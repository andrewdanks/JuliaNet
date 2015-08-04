immutable type Batch
    input::InputTensor
    target_output::T_TENSOR
    target_classes::Vector

    Batch(input::InputTensor, target_output::T_TENSOR) = new(input, target_output)
    Batch(input::InputTensor, target_output::T_TENSOR, target_classes::Vector) = new(input, target_output, target_classes)
end


function Base.size(batch::Batch)
    batch.input.batch_size
end


function get_chunk_from_batch(batch::Batch, range::UnitRange{T_INT})
    Batch(
        InputTensor(input_range(batch.input, range)),
        batch.target_output[:, range],
        batch.target_classes[range]
    )
end


function get_batch_chunks(batches::Vector{Batch})
    batch_chunks = Vector[]
    for batch in batches
        batch_size = size(batch)
        half_batch_size = int(batch_size/2)
        chunk1 = get_chunk_from_batch(batch, UnitRange(1, half_batch_size))
        chunk2 = get_chunk_from_batch(batch, UnitRange(1+half_batch_size, batch_size))
        push!(batch_chunks, [chunk1, chunk2])
    end
    batch_chunks
end