# Add timestamps to logger
const time_format = "HH:MM:SS.sss"
timestamp_logger(logger) = TransformerLogger(logger) do log 
  merge(log, (; message = "$(format(now(), time_format)) $(log.message)"))
end

MOCNeutronTransport_logger_timestamps_on = false
function add_timestamps_to_logger()
    if !MOCNeutronTransport_logger_timestamps_on
        logger = global_logger()
        logger |> timestamp_logger |> global_logger
        global MOCNeutronTransport_logger_timestamps_on = true
    end 
end
