export add_timestamps_to_log

const logger_time_format = "HH:MM:SS.sss"

LOG_TIMESTAMPS_ON::Bool = false

# Add timestamps to logger
timestamp_logger(logger) = TransformerLogger(logger) do log
    merge(log, (; message = "$(format(now(), logger_time_format)) $(log.message)"))
end

function add_timestamps_to_log()
    if !LOG_TIMESTAMPS_ON
        logger = global_logger()
        logger |> timestamp_logger |> global_logger
        global LOG_TIMESTAMPS_ON = true
    end
    return LOG_TIMESTAMPS_ON
end
