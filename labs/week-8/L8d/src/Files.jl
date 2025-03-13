function _jld2(path::String)::Dict{String,Any}
    return load(path);
end

# -- PUBLIC METHODS BELOW HERE ---------------------------------------------------------------------------------------- #
MyInSampleSIMParameters() = _jld2(joinpath(_PATH_TO_DATA, "SIMs-SP500-01-03-18-to-12-29-23.jld2"));
MyOutOfSamepleMarketDataSet() = _jld2(joinpath(_PATH_TO_DATA, "SP500-Daily-OHLC-1-3-2024-to-12-04-2024.jld2"));
# --- PUBLIC METHODS ABOVE HERE --------------------------------------------------------------------------------------- #