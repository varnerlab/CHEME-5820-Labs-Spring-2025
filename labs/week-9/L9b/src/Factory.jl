# -- PUBLIC METHODS BELOW HERE ---------------------------------------------------------------------------------------- #
"""
Fill me in.
"""
function build(modeltype::Type{MyClassicalHopfieldNetworkModel}, data::NamedTuple)::MyClassicalHopfieldNetworkModel

    # initialize -
    model = modeltype();
    linearimagecollection = data.memories;
    number_of_rows, number_of_cols = size(linearimagecollection);
    W = zeros(Float32, number_of_rows, number_of_rows);
    b = zeros(Float32, number_of_rows);
    L = -1.0; # lower bound
    U = 1.0; # upper bound

    # compute the W -
    for j ∈ 1:number_of_cols
        Y = ⊗(linearimagecollection[:,j], linearimagecollection[:,j]); # compute the outer product -
        W += Y; # update the W -
    end
    WN = (1/number_of_cols)*W; # normalize the W -

    # generate a random bias vector -
    for i ∈ 1:number_of_rows
        f = rand();
        b[i] = f*U+(1-f)*L;
    end
    
    # compute the energy dictionary -
    energy = Dict{Int64, Float32}();
    for i ∈ 1:number_of_cols
        energy[i] = _energy(linearimagecollection[:,i], WN, b);
    end

    # add data to the model -
    model.W = WN;
    model.b = b;
    model.energy = energy;

    # return -
    return model;
end
# --- PUBLIC METHODS ABOVE HERE --------------------------------------------------------------------------------------- #