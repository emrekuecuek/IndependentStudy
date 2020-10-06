
using HDF5
# using JLD
using Knet
# using IterTools: ncycle

function read_data(datafile, namelist, dims)
    sz = length(namelist)
    data = Array(zeros(Float32, dims..., sz))
    fid = h5open(datafile)
    @inbounds for i = 1:sz
        input = read(fid, convert(String, namelist[i]))
        data[:, :, :, :, i] = input
    end
    close(fid)
    data_gpu = ka(data)
    return data_gpu
end

function read_namelist(file)#y
    #Reading the list of protein names from the given file name
    text = readlines(file)
    text = [split(i) for i in text]
    labels = zeros(Int32,length(text))
    for i in 1:length(text)
        labels[i] = parse(Int32, text[i][2]) + 1 # 1 ->0, 2->1
    end
    return [i[1] for i in text], labels
end

# Convolutional layer:
struct Conv; w; b; f; end
(c::Conv)(x) = c.f.(pool(conv4(c.w, x) .+ c.b))
Conv(w1,w2,w3,cx,cy,f=relu) = Conv(param(w1,w2,w3,cx,cy), param0(1,1,1,cy,1), f)

# Dense layer:
struct Dense; w; b; f; end
(d::Dense)(x) = d.f.(d.w * mat(x) .+ d.b)
Dense(i::Int,o::Int,f=relu) = Dense(param(o,i), param0(o), f)

# A chain of layers and a loss function:
struct Chain; layers; end
(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)
(c::Chain)(x,y) = nll(c(x),y)

LeNet = Chain((Conv(5,5,5,40,50), Conv(5,5,5,50,60), Dense(1620,920), Dense(920,2,identity)));

data_path = "/scratch/users/ekucuk19/workfolder/Cosbilab/emre_deepinterface/interactome_data/data/deepinter_data.h5";
vallist, vallabels = read_namelist("/scratch/users/ekucuk19/workfolder/Cosbilab/emre_deepinterface/interactome_data/data/val_scrambled_2.txt");
trnlist, trnlabels = read_namelist("/scratch/users/ekucuk19/workfolder/Cosbilab/emre_deepinterface/interactome_data/data/train_scrambled_2.txt");
# tstlist, tstlabels = read_namelist("/scratch/users/ekucuk19/workfolder/Cosbilab/emre_deepinterface/interactome_data/data/test_scrambled_2.txt");

dtrn = minibatch(trnlist, trnlabels, 100)
dval = minibatch(vallist, vallabels, 100)
# dtst = minibatch(tstlist, tstlabels, 100)

val_acc_history = Array{Float64,1}()
trn_acc_history = Array{Float64,1}()
for (index,(names,labels)) in enumerate(dtrn)
    data = read_data(data_path,names,(25,25,25,40));
    adam!(LeNet, [(data, labels)])
    if index%100 == 0
        println("Validation starts.")
        counter = 0;
        val_acc = 0;
        for (names,labels) in dval
            val_data = read_data(data_path,names,(25,25,25,40))
            counter += 1
            val_acc += accuracy(LeNet(val_data),labels)
        end     
        avg_val_acc = val_acc/counter
        push!(val_acc_history, avg_val_acc)
        println(avg_val_acc)
        println("End of validation.")
    end
    minibatch_err = accuracy(LeNet(data),labels)
    push!(trn_acc_history, minibatch_err)
    println("Iter:$(index)/2555, Progress:%$((index*100)/2555), Iter Acc:$(minibatch_err)")
end

Knet.@save "model_adam.jld2" LeNet val_acc_history trn_acc_history

trn_acc_history
