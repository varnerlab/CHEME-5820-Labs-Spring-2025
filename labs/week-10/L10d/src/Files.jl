function parser(filepath::String; numberoffields::Int = 256)
    
    # initialize -
    records = Dict{Int, Array{Float64, 1}}();
    labels = Array{Int, 1}();
    
    # open the file, process each line -
    linecounter = 1;
    open(filepath, "r") do io # open a stream to the file
        for line ∈ eachline(io)
            
            # fields -
            fields = split(line, " ");
            y = parse(Int, fields[1]); # first field is the Int label
            push!(labels, y); # store the label in the labels array
            # println("label: ", y);
            
            # split around the : character -
            record = Dict{Int, Float64}();
            for field ∈ fields[2:end]

                if (isempty(field) == false)
                    # split the field around the : character -
                    key, value = split(field, ":");
                    key = parse(Int, key);
                    value = parse(Float64, value);
                    record[key] = value;
                end
            end

            # store the record -
            records[linecounter] = [record[i] for i ∈ 1:numberoffields];
            linecounter += 1;
        end
    end


    return (labels, records);
end