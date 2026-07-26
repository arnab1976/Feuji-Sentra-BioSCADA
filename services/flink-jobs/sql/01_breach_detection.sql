-- =====================================================================
-- BioSCADA AI — Phase 1: Flink SQL continuous query
--
-- This is the RELATIONAL-PLAN decomposition: one declarative statement
-- that Flink breaks into a distributed operator graph:
--   Source -> Watermark -> Window -> Aggregate(features) -> UDF score
--   -> Filter/HAVING -> Sink
--
-- The sink emits the enriched "breach + why" event. That structured
-- payload is the hand-off boundary; SEMANTIC (RAG) decomposition happens
-- downstream in the agent layer, never here.
--
-- Run with:  sql-client.sh -f 01_breach_detection.sql
-- =====================================================================

SET 'pipeline.name' = 'bioscada-breach-detection';
SET 'execution.runtime-mode' = 'streaming';
SET 'table.exec.source.idle-timeout' = '10s';
SET 'parallelism.default' = '2';

-- ---------------------------------------------------------------------
-- SOURCE — raw SCADA telemetry from Kafka
-- Nested `features` map holds the independent variables (PdM features).
-- Event-time + watermark makes windowing correct under out-of-order data.
-- ---------------------------------------------------------------------
CREATE TABLE scada_telemetry (
    `param`     STRING,
    `asset`     STRING,
    `value`     DOUBLE,
    `unit`      STRING,
    `zone`      STRING,
    `batch_id`  STRING,
    `molecule`  STRING,
    `ts`        TIMESTAMP_LTZ(3),
    `features`  MAP<STRING, DOUBLE>,
    WATERMARK FOR `ts` AS `ts` - INTERVAL '5' SECOND
) WITH (
    'connector'                             = 'kafka',
    'topic'                                 = 'scada.telemetry',
    'properties.bootstrap.servers'          = 'kafka:29092',
    'properties.group.id'                   = 'bioscada-flink',
    'scan.startup.mode'                     = 'latest-offset',
    'format'                                = 'json',
    'json.timestamp-format.standard'        = 'ISO-8601',
    'json.ignore-parse-errors'              = 'true'
);

-- ---------------------------------------------------------------------
-- SINK — enriched breach events consumed by the agent / RAG layer
-- ---------------------------------------------------------------------
CREATE TABLE breach_events (
    `event_id`     STRING,
    `param`        STRING,
    `asset`        STRING,
    `batch_id`     STRING,
    `window_start` TIMESTAMP_LTZ(3),
    `window_end`   TIMESTAMP_LTZ(3),
    `v_avg`        DOUBLE,
    `v_min`        DOUBLE,
    `v_max`        DOUBLE,
    `v_std`        DOUBLE,
    `v_delta`      DOUBLE,
    `n_rows`       BIGINT,
    `zone`         STRING,
    `p_breach`     DOUBLE,
    `top_driver`   STRING,
    `emitted_at`   TIMESTAMP_LTZ(3),
    PRIMARY KEY (`event_id`) NOT ENFORCED
) WITH (
    'connector'                    = 'upsert-kafka',
    'topic'                        = 'breach.events',
    'properties.bootstrap.servers' = 'kafka:29092',
    'key.format'                   = 'json',
    'value.format'                 = 'json'
);

-- ---------------------------------------------------------------------
-- Optional sink — every windowed feature row, for the feature store /
-- model training set (Phase 2 consumes this).
-- ---------------------------------------------------------------------
CREATE TABLE feature_rows (
    `param`        STRING,
    `window_start` TIMESTAMP_LTZ(3),
    `v_avg`        DOUBLE,
    `v_std`        DOUBLE,
    `v_delta`      DOUBLE,
    `zone`         STRING,
    `p_breach`     DOUBLE
) WITH (
    'connector'                    = 'kafka',
    'topic'                        = 'features.windowed',
    'properties.bootstrap.servers' = 'kafka:29092',
    'format'                       = 'json'
);

-- ---------------------------------------------------------------------
-- THE CONTINUOUS QUERY
--
-- Operator plan Flink derives from this single statement:
--   1. Source        : consume scada.telemetry (unbounded)
--   2. Watermark     : event-time, 5s out-of-orderness
--   3. Window        : 10s tumbling, keyed by param
--   4. Aggregate     : v_avg / v_std / v_delta  <- independent variables
--   5. Scalar UDF    : PDM_SCORE(...)           <- model inference in-stream
--   6. Filter (HAVING): keep only breach candidates
--   7. Sink          : emit "breach + why"
-- ---------------------------------------------------------------------
INSERT INTO breach_events
SELECT
    CONCAT('evt-', `param`, '-', CAST(UNIX_TIMESTAMP(CAST(window_start AS STRING)) AS STRING)) AS `event_id`,
    `param`,
    MAX(`asset`)                                            AS `asset`,
    MAX(`batch_id`)                                         AS `batch_id`,
    window_start,
    window_end,
    ROUND(AVG(`value`), 4)                                  AS `v_avg`,
    MIN(`value`)                                            AS `v_min`,
    MAX(`value`)                                            AS `v_max`,
    ROUND(STDDEV_POP(`value`), 4)                           AS `v_std`,
    ROUND(MAX(`value`) - MIN(`value`), 4)                   AS `v_delta`,
    COUNT(*)                                                AS `n_rows`,
    -- worst zone observed in the window
    MAX(`zone`)                                             AS `zone`,
    -- in-stream model inference (Python UDF, registered by the job)
    ROUND(PDM_SCORE(
        `param`,
        AVG(`value`),
        STDDEV_POP(`value`),
        MAX(`value`) - MIN(`value`)
    ), 4)                                                   AS `p_breach`,
    -- the "why": which independent variable dominates
    TOP_DRIVER(`param`, AVG(`value`))                       AS `top_driver`,
    CURRENT_TIMESTAMP                                       AS `emitted_at`
FROM TABLE(
    TUMBLE(TABLE scada_telemetry, DESCRIPTOR(`ts`), INTERVAL '10' SECOND)
)
GROUP BY `param`, window_start, window_end
HAVING
    -- breach candidates only: either the model is worried, or we already
    -- left the control band. Everything else is dropped here.
    PDM_SCORE(`param`, AVG(`value`), STDDEV_POP(`value`), MAX(`value`) - MIN(`value`)) > 0.30
    OR MAX(`zone`) <> 'control';
