"""
    confusion(actual::Array{Int64,1}, model::Array{Int64,1}) -> Array{Int64,2}

The function computes the confusion matrix for the classification model.

### Arguments
- `actual::Array{<:Number,1}`: the actual labels.
- `model::Array{<:Number,1}`: the model estimated labels.

### Returns
- a 2x2 confusion matrix.
"""
function confusion(actual::Array{<:Number,1}, model::Array{<:Number,1})::Array{Int64,2}

    # compute the confusion matrix -
    number_of_test_examples = size(actual,1);
    confusion_matrix = zeros(Int64,2,2);
    y = actual;
    ŷ = model;
    
    # True positive: TP (cancer)
    counter = 0;
    for i ∈ 1:number_of_test_examples
        if (y[i] == 1 && ŷ[i] == 1)
            counter+=1;
        end
    end
    confusion_matrix[1,1] = counter;

    # False negative: FN
    counter = 0;
    for i ∈ 1:number_of_test_examples
        if (y[i] == 1 && ŷ[i] == -1)
            counter+=1;
        end
    end
    confusion_matrix[1,2] = counter;

    # False position: FP
    counter = 0;
    for i ∈ 1:number_of_test_examples
        if (y[i] == -1 && ŷ[i] == 1)
            counter+=1;
        end
    end
    confusion_matrix[2,1] = counter;

    # True negative: TN
    counter = 0;
    for i ∈ 1:number_of_test_examples
        if (y[i] == -1 && ŷ[i] == -1)
            counter+=1;
        end
    end
    confusion_matrix[2,2] = counter;

    # return -
    confusion_matrix
end

# --- PUBLIC API ABOVE HERE --------------------------------------------------------------------------------------- #