# Add timestamps to logger
const date_format = "HH:MM:SS.sss"
timestamp_logger(logger) = TransformerLogger(logger) do log 
  merge(log, (; message = "$(format(now(), date_format)) $(log.message)"))
end

MOCNeutronTransport_timestamps_on = false
function log_timestamps()
    if !MOCNeutronTransport_timestamps_on
        logger = global_logger()
        logger |> timestamp_logger |> global_logger
        global MOCNeutronTransport_timestamps_on = true
    end 
end
