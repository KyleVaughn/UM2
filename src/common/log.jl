export add_timestamps_to_logger

const logger_time_format = "HH:MM:SS.sss"

# Add timestamps to logger
function timestamp_logger(logger)
    TransformerLogger(logger) do log
        return merge(log,
                     (; message = "$(format(now(), logger_time_format)) $(log.message)"))
    end
end

logger_timestamps_on::Bool = false
function add_timestamps_to_logger()
    if !logger_timestamps_on
        logger = global_logger()
        logger |> timestamp_logger |> global_logger
        global logger_timestamps_on = true
    end
    return logger_timestamps_on
end
