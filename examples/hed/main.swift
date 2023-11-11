import Diffusion
import Foundation
import NNC
import NNCPythonConversion
import PNG
import PythonKit


let torch = Python.import("torch")
let getopt = Python.import("getopt")
let numpy = Python.import("numpy")
let Image = Python.import("PIL.Image")
let sys = Python.import("sys")

let customPath = "/home/wlin1/drawThings/swift-diffusion/examples/hed/pytorch-hed"
let imagePath = "/home/wlin1/drawThings/swift-diffusion/examples/hed/pytorch-hed/images/rsz_sample.png"

sys.path.append(customPath)
let run = Python.import("run")
let netNetwork = run.Network().cuda().eval()
// print(netNetwork)
run.test()
let state_dict = netNetwork.state_dict()
// print(state_dict.keys())

var tenInput = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(Image.open(imagePath)).transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
tenInput = tenInput * 255.0
tenInput = tenInput - torch.tensor(data:[104.00698793, 116.66876762, 122.67891434], dtype:tenInput.dtype, device:tenInput.device).view(1, 3, 1, 1)

func vggConvLayer(outputChannels:Int, convLayers: Int,  prefix: String) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let maxPool = MaxPool(filterSize: [2, 2], hint: Hint(stride: [2, 2]))
  var out = maxPool(x)
  
  var layerConv2dArray =  [Convolution]()
  
  for _ in 0..<convLayers {
    let layerConv2d = Convolution(
    groups: 1, filters: outputChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OHWI)
    out = layerConv2d(out)
    out = ReLU()(out)
    layerConv2dArray.append(layerConv2d)
  }

  let reader: (PythonObject) -> Void = { state_dict in
    for i in 0..<convLayers {
        let index = i * 2 + 1
        let layerConv2d_weight = state_dict["\(prefix).\(index).weight"].type(torch.float).cpu().numpy()
        let layerConv2d_bias = state_dict["\(prefix).\(index).bias"].type(torch.float).cpu().numpy()
        layerConv2dArray[i].weight.copy(from: try! Tensor<Float>(numpy: layerConv2d_weight))
        layerConv2dArray[i].bias.copy(from: try! Tensor<Float>(numpy: layerConv2d_bias))
    }
  }

  return (reader, Model([x], [out]))
}

func HedModelVggEmbeds(inputWidth: Int, inputHeight: Int) -> ((PythonObject) -> Void, Model) {
  let x = Input()

  let vggOneInLayerConv2d = Convolution(
    groups: 1, filters: 64, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OHWI)
  var vggOneEmbeds = vggOneInLayerConv2d(x)
  vggOneEmbeds = ReLU()(vggOneEmbeds)

  let vggOneOutLayerConv2d = Convolution(
    groups: 1, filters: 64, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OHWI)
  vggOneEmbeds = vggOneOutLayerConv2d(vggOneEmbeds)
  vggOneEmbeds = ReLU()(vggOneEmbeds)

  let (reader2, vggTwo) = vggConvLayer(outputChannels:128, convLayers: 2, prefix: "netVggTwo")
  let vggTwoEmbeds = vggTwo(vggOneEmbeds)
  
  let (reader3, vggThr) = vggConvLayer(outputChannels:256, convLayers: 3, prefix: "netVggThr")
  let vggThrEmbeds = vggThr(vggTwoEmbeds)

  let (reader4, vggFour) = vggConvLayer(outputChannels:512, convLayers: 3, prefix: "netVggFou")
  let vggFouEmbeds = vggFour(vggThrEmbeds)

  let (reader5, vggFiv) = vggConvLayer(outputChannels:512, convLayers: 3, prefix: "netVggFiv")
  let vggFiveEmbeds = vggFiv(vggFouEmbeds)

  let netScoreOne = Convolution(
    groups: 1, filters: 1, filterSize: [1, 1],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])), format: .OHWI)
 
  let netScoreTwo = Convolution(
    groups: 1, filters: 1, filterSize: [1, 1],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])), format: .OHWI)
 
  let netScoreThr = Convolution(
    groups: 1, filters: 1, filterSize: [1, 1],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])), format: .OHWI)

    let netScoreFou = Convolution(
    groups: 1, filters: 1, filterSize: [1, 1],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])), format: .OHWI)

    let netScoreFiv = Convolution(
    groups: 1, filters: 1, filterSize: [1, 1],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])), format: .OHWI)

   let scaledNetScoreOneEmbeds = Upsample(.bilinear, widthScale: 1, heightScale: 1)(netScoreOne(vggOneEmbeds))
   let scaledNetScoreTwoEmbeds = Upsample(.bilinear, widthScale: 2, heightScale: 2)(netScoreTwo(vggTwoEmbeds))
   let scaledNetScoreThrEmbeds = Upsample(.bilinear, widthScale: 4, heightScale: 4)(netScoreThr(vggThrEmbeds))
   let scaledNetScoreFouEmbeds = Upsample(.bilinear, widthScale: 8, heightScale: 8)(netScoreFou(vggFouEmbeds))
   let scaledNetScoreFivEmbeds = Upsample(.bilinear, widthScale: 16, heightScale: 16)(netScoreFiv(vggFiveEmbeds))
   let mergedVggEmbedings = Functional.concat(axis: 3, scaledNetScoreOneEmbeds, scaledNetScoreTwoEmbeds, scaledNetScoreThrEmbeds, scaledNetScoreFouEmbeds, scaledNetScoreFivEmbeds)

  let netcombineLayerConv2d = Convolution(
    groups: 1, filters: 1, filterSize: [1, 1],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])), format: .OHWI)
  var out = netcombineLayerConv2d(mergedVggEmbedings)
  out = Sigmoid()(out)

  let readerExternal: (PythonObject) -> Void = { state_dict in
    reader2(state_dict)
    reader3(state_dict)
    reader4(state_dict)
    reader5(state_dict)

    let netScoreOne_weight = state_dict["netScoreOne.weight"].type(torch.float).cpu().numpy()
    let netScoreOne_bias = state_dict["netScoreOne.bias"].type(torch.float).cpu().numpy()
    netScoreOne.weight.copy(from: try! Tensor<Float>(numpy: netScoreOne_weight))
    netScoreOne.bias.copy(from: try! Tensor<Float>(numpy: netScoreOne_bias))
    
    let netScoreTwo_weight = state_dict["netScoreTwo.weight"].type(torch.float).cpu().numpy()
    let netScoreTwo_bias = state_dict["netScoreTwo.bias"].type(torch.float).cpu().numpy()
    netScoreTwo.weight.copy(from: try! Tensor<Float>(numpy: netScoreTwo_weight))
    netScoreTwo.bias.copy(from: try! Tensor<Float>(numpy: netScoreTwo_bias))

    let netScoreThr_weight = state_dict["netScoreThr.weight"].type(torch.float).cpu().numpy()
    let netScoreThr_bias = state_dict["netScoreThr.bias"].type(torch.float).cpu().numpy()
    netScoreThr.weight.copy(from: try! Tensor<Float>(numpy: netScoreThr_weight))
    netScoreThr.bias.copy(from: try! Tensor<Float>(numpy: netScoreThr_bias))

    let netScoreFou_weight = state_dict["netScoreFou.weight"].type(torch.float).cpu().numpy()
    let netScoreFou_bias = state_dict["netScoreFou.bias"].type(torch.float).cpu().numpy()
    netScoreFou.weight.copy(from: try! Tensor<Float>(numpy: netScoreFou_weight))
    netScoreFou.bias.copy(from: try! Tensor<Float>(numpy: netScoreFou_bias))

    let netScoreFiv_weight = state_dict["netScoreFiv.weight"].type(torch.float).cpu().numpy()
    let netScoreFiv_bias = state_dict["netScoreFiv.bias"].type(torch.float).cpu().numpy()
    netScoreFiv.weight.copy(from: try! Tensor<Float>(numpy: netScoreFiv_weight))
    netScoreFiv.bias.copy(from: try! Tensor<Float>(numpy: netScoreFiv_bias))

    let netcombineLayerConv2d_weight = state_dict["netCombine.0.weight"].type(torch.float).cpu().numpy()
    let netcombineLayerConv2d_bias = state_dict["netCombine.0.bias"].type(torch.float).cpu().numpy()
    netcombineLayerConv2d.weight.copy(from: try! Tensor<Float>(numpy: netcombineLayerConv2d_weight))
    netcombineLayerConv2d.bias.copy(from: try! Tensor<Float>(numpy: netcombineLayerConv2d_bias))

    let vggOneInLayerConv2d_weight = state_dict["netVggOne.0.weight"].type(torch.float).cpu().numpy()
    let vggOneInLayerConv2d_bias = state_dict["netVggOne.0.bias"].type(torch.float).cpu().numpy()
    vggOneInLayerConv2d.weight.copy(from: try! Tensor<Float>(numpy: vggOneInLayerConv2d_weight))
    vggOneInLayerConv2d.bias.copy(from: try! Tensor<Float>(numpy: vggOneInLayerConv2d_bias))
    let vggOneOutLayerConv2d_weight = state_dict["netVggOne.2.weight"].type(torch.float).cpu().numpy()
    let vggOneOutLayerConv2d_bias = state_dict["netVggOne.2.bias"].type(torch.float).cpu().numpy()
    vggOneOutLayerConv2d.weight.copy(from: try! Tensor<Float>(numpy: vggOneOutLayerConv2d_weight))
    vggOneOutLayerConv2d.bias.copy(from: try! Tensor<Float>(numpy: vggOneOutLayerConv2d_bias))
  }

  return (readerExternal, Model([x], [out]))
}


let graph = DynamicGraph()
graph.withNoGrad {
  print("the before transpose:")
  debugPrint(tenInput.type(torch.float).cpu().numpy())
  let tenInput = tenInput.type(torch.float).cpu().numpy().transpose(0, 2, 3, 1)
  print("after transpose:")
  debugPrint(tenInput)
  let tenInputTensor = graph.variable(try! Tensor<Float>(numpy: tenInput)).toGPU(0)
  let shape = tenInputTensor.shape
  let inputWidth = shape[2]
  let inputHeight = shape[3]
  NC

  let (hedVggReader,hedModelVgg) = HedModelVggEmbeds(inputWidth:inputWidth, inputHeight:inputHeight)
  hedModelVgg.compile(inputs: tenInputTensor)
  hedVggReader(state_dict)
  // graph.openStore("/home/wlin1/drawThings/swift-diffusion/examples/hed/hed.ckpt") {
  //   $0.read("hed_vgg", model: hedModelVgg)
  // }
  

  let modeulResult = hedModelVgg(inputs: tenInputTensor)[0].as(of: Float.self)


  print("swift merged result:")
  debugPrint(modeulResult)

  var result = modeulResult * 255.0
    print("swift merged result:")
  debugPrint(result)
  result = result.reshaped(.NC(inputWidth, inputHeight))
  let p = result.rawValue.toCPU()
  debugPrint(p)

    // PIL.Image.fromarray((tenOutput.clip(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0] * 255.0).astype(numpy.uint8)).save(arguments_strOut)

  let t = Tensor<Float>(from: p ).makeNumpyArray().clip(0.0, 255.0)
  let r = t.astype(numpy.uint8)
  debugPrint(r)
  print(r.shape)
  Image.fromarray(r).save("/home/wlin1/drawThings/swift-diffusion/examples/hed/pytorch-hed/out2.png")
}

/*
func vggConvLayer(outputChannels:Int, convLayers: Int,  prefix: String) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let maxPool = MaxPool(filterSize: [2, 2], hint: Hint(stride: [2, 2]))
  var out = maxPool(x)
  
  var layerConv2dArray =  [Convolution]()
  
  for _ in 0..<convLayers {
    let layerConv2d = Convolution(
    groups: 1, filters: outputChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
    out = layerConv2d(out)
    out = ReLU()(out)
    layerConv2dArray.append(layerConv2d)
  }

  let reader: (PythonObject) -> Void = { state_dict in
    for i in 0..<convLayers {
        let index = i * 2 + 1
        let layerConv2d_weight = state_dict["\(prefix).\(index).weight"].type(torch.float).cpu().numpy()
        let layerConv2d_bias = state_dict["\(prefix).\(index).bias"].type(torch.float).cpu().numpy()
        layerConv2dArray[i].weight.copy(from: try! Tensor<Float>(numpy: layerConv2d_weight))
        layerConv2dArray[i].bias.copy(from: try! Tensor<Float>(numpy: layerConv2d_bias))
    }
  }

  return (reader, Model([x], [out]))
}

func HedModelVggEmbeds(inputWidth: Int, inputHeight: Int) -> ((PythonObject) -> Void, Model) {
  let x = Input()

  let vggOneInLayerConv2d = Convolution(
    groups: 1, filters: 64, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var vggOneEmbeds = vggOneInLayerConv2d(x)
  vggOneEmbeds = ReLU()(vggOneEmbeds)

  let vggOneOutLayerConv2d = Convolution(
    groups: 1, filters: 64, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  vggOneEmbeds = vggOneOutLayerConv2d(vggOneEmbeds)
  vggOneEmbeds = ReLU()(vggOneEmbeds)

  let (reader2, vggTwo) = vggConvLayer(outputChannels:128, convLayers: 2, prefix: "netVggTwo")
  let vggTwoEmbeds = vggTwo(vggOneEmbeds)
  
  let (reader3, vggThr) = vggConvLayer(outputChannels:256, convLayers: 3, prefix: "netVggThr")
  let vggThrEmbeds = vggThr(vggTwoEmbeds)

  let (reader4, vggFour) = vggConvLayer(outputChannels:512, convLayers: 3, prefix: "netVggFou")
  let vggFouEmbeds = vggFour(vggThrEmbeds)

  let (reader5, vggFiv) = vggConvLayer(outputChannels:512, convLayers: 3, prefix: "netVggFiv")
  let vggFiveEmbeds = vggFiv(vggFouEmbeds)

  let netScoreOne = Convolution(
    groups: 1, filters: 1, filterSize: [1, 1],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])))
 
  let netScoreTwo = Convolution(
    groups: 1, filters: 1, filterSize: [1, 1],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])))
 
  let netScoreThr = Convolution(
    groups: 1, filters: 1, filterSize: [1, 1],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])))

    let netScoreFou = Convolution(
    groups: 1, filters: 1, filterSize: [1, 1],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])))

    let netScoreFiv = Convolution(
    groups: 1, filters: 1, filterSize: [1, 1],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])))

   let scaledNetScoreOneEmbeds = Upsample(.bilinear, widthScale: 1, heightScale: 1)(netScoreOne(vggOneEmbeds))
   let scaledNetScoreTwoEmbeds = Upsample(.bilinear, widthScale: 2, heightScale: 2)(netScoreTwo(vggTwoEmbeds))
   let scaledNetScoreThrEmbeds = Upsample(.bilinear, widthScale: 4, heightScale: 4)(netScoreThr(vggThrEmbeds))
   let scaledNetScoreFouEmbeds = Upsample(.bilinear, widthScale: 8, heightScale: 8)(netScoreFou(vggFouEmbeds))
   let scaledNetScoreFivEmbeds = Upsample(.bilinear, widthScale: 16, heightScale: 16)(netScoreFiv(vggFiveEmbeds))
   let mergedVggEmbedings = Functional.concat(axis: 1, scaledNetScoreOneEmbeds, scaledNetScoreTwoEmbeds, scaledNetScoreThrEmbeds, scaledNetScoreFouEmbeds, scaledNetScoreFivEmbeds)

  let netcombineLayerConv2d = Convolution(
    groups: 1, filters: 1, filterSize: [1, 1],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])))
  var out = netcombineLayerConv2d(mergedVggEmbedings)
  out = Sigmoid()(out)

  let readerExternal: (PythonObject) -> Void = { state_dict in
    reader2(state_dict)
    reader3(state_dict)
    reader4(state_dict)
    reader5(state_dict)

    let netScoreOne_weight = state_dict["netScoreOne.weight"].type(torch.float).cpu().numpy()
    let netScoreOne_bias = state_dict["netScoreOne.bias"].type(torch.float).cpu().numpy()
    netScoreOne.weight.copy(from: try! Tensor<Float>(numpy: netScoreOne_weight))
    netScoreOne.bias.copy(from: try! Tensor<Float>(numpy: netScoreOne_bias))
    
    let netScoreTwo_weight = state_dict["netScoreTwo.weight"].type(torch.float).cpu().numpy()
    let netScoreTwo_bias = state_dict["netScoreTwo.bias"].type(torch.float).cpu().numpy()
    netScoreTwo.weight.copy(from: try! Tensor<Float>(numpy: netScoreTwo_weight))
    netScoreTwo.bias.copy(from: try! Tensor<Float>(numpy: netScoreTwo_bias))

    let netScoreThr_weight = state_dict["netScoreThr.weight"].type(torch.float).cpu().numpy()
    let netScoreThr_bias = state_dict["netScoreThr.bias"].type(torch.float).cpu().numpy()
    netScoreThr.weight.copy(from: try! Tensor<Float>(numpy: netScoreThr_weight))
    netScoreThr.bias.copy(from: try! Tensor<Float>(numpy: netScoreThr_bias))

    let netScoreFou_weight = state_dict["netScoreFou.weight"].type(torch.float).cpu().numpy()
    let netScoreFou_bias = state_dict["netScoreFou.bias"].type(torch.float).cpu().numpy()
    netScoreFou.weight.copy(from: try! Tensor<Float>(numpy: netScoreFou_weight))
    netScoreFou.bias.copy(from: try! Tensor<Float>(numpy: netScoreFou_bias))

    let netScoreFiv_weight = state_dict["netScoreFiv.weight"].type(torch.float).cpu().numpy()
    let netScoreFiv_bias = state_dict["netScoreFiv.bias"].type(torch.float).cpu().numpy()
    netScoreFiv.weight.copy(from: try! Tensor<Float>(numpy: netScoreFiv_weight))
    netScoreFiv.bias.copy(from: try! Tensor<Float>(numpy: netScoreFiv_bias))

    let netcombineLayerConv2d_weight = state_dict["netCombine.0.weight"].type(torch.float).cpu().numpy()
    let netcombineLayerConv2d_bias = state_dict["netCombine.0.bias"].type(torch.float).cpu().numpy()
    netcombineLayerConv2d.weight.copy(from: try! Tensor<Float>(numpy: netcombineLayerConv2d_weight))
    netcombineLayerConv2d.bias.copy(from: try! Tensor<Float>(numpy: netcombineLayerConv2d_bias))

    let vggOneInLayerConv2d_weight = state_dict["netVggOne.0.weight"].type(torch.float).cpu().numpy()
    let vggOneInLayerConv2d_bias = state_dict["netVggOne.0.bias"].type(torch.float).cpu().numpy()
    vggOneInLayerConv2d.weight.copy(from: try! Tensor<Float>(numpy: vggOneInLayerConv2d_weight))
    vggOneInLayerConv2d.bias.copy(from: try! Tensor<Float>(numpy: vggOneInLayerConv2d_bias))
    let vggOneOutLayerConv2d_weight = state_dict["netVggOne.2.weight"].type(torch.float).cpu().numpy()
    let vggOneOutLayerConv2d_bias = state_dict["netVggOne.2.bias"].type(torch.float).cpu().numpy()
    vggOneOutLayerConv2d.weight.copy(from: try! Tensor<Float>(numpy: vggOneOutLayerConv2d_weight))
    vggOneOutLayerConv2d.bias.copy(from: try! Tensor<Float>(numpy: vggOneOutLayerConv2d_bias))
  }

  return (readerExternal, Model([x], [out]))
}


let graph = DynamicGraph()
graph.withNoGrad {
  let tenInput = tenInput.type(torch.float).cpu().numpy()
  let tenInputTensor = graph.variable(try! Tensor<Float>(numpy: tenInput)).toGPU(0)
  let shape = tenInputTensor.shape
  let inputWidth = shape[2]
  let inputHeight = shape[3]

  let (hedVggReader,hedModelVgg) = HedModelVggEmbeds(inputWidth:inputWidth, inputHeight:inputHeight)
  hedModelVgg.compile(inputs: tenInputTensor)
  graph.openStore("/home/wlin1/drawThings/swift-diffusion/examples/hed/hed.ckpt") {
    $0.read("hed_vgg", model: hedModelVgg)
  }
  
  print("the InputTensor:")
  debugPrint(tenInputTensor)
  let modeulResult = hedModelVgg(inputs: tenInputTensor)[0].as(of: Float.self)


  print("swift merged result:")
  debugPrint(modeulResult)

  var result = modeulResult * 255.0
    print("swift merged result:")
  debugPrint(result)
  result = result.reshaped(.NC(inputWidth, inputHeight))
  let p = result.rawValue.toCPU()
  debugPrint(p)

    // PIL.Image.fromarray((tenOutput.clip(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0] * 255.0).astype(numpy.uint8)).save(arguments_strOut)

  let t = Tensor<Float>(from: p ).makeNumpyArray().clip(0.0, 255.0)
  let r = t.astype(numpy.uint8)
  debugPrint(r)
  print(r.shape)
  Image.fromarray(r).save("/home/wlin1/drawThings/swift-diffusion/examples/hed/pytorch-hed/out2.png")
}
*/
/*
let torch = Python.import("torch")
let getopt = Python.import("getopt")
let numpy = Python.import("numpy")
let Image = Python.import("PIL.Image")
let sys = Python.import("sys")

let customPath = "/home/wlin1/drawThings/swift-diffusion/examples/hed/pytorch-hed"
let imagePath = "/home/wlin1/drawThings/swift-diffusion/examples/hed/pytorch-hed/images/sample.png"

sys.path.append(customPath)
let run = Python.import("run")
let netNetwork = run.Network().cuda().eval()
// print(netNetwork)
run.test()
// let state_dict = netNetwork.state_dict()
// print(state_dict.keys())

var tenInput = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(Image.open(imagePath)).transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))

// tenInput = tenInput * 255.0
// print("print(tenInput * 255.0)")
// print(tenInput * 255.0)

tenInput = tenInput * 255.0 - torch.tensor(data:[ 122.67891434, 116.66876762, 104.00698793], dtype:tenInput.dtype, device:tenInput.device).view(1, 3, 1, 1)
// print("print(tenInput * 255.0 -)")
// print(tenInput)

func vggConvLayer(outputChannels:Int, convLayers: Int) -> Model {
  let x = Input()
  let maxPool = MaxPool(filterSize: [2, 2], hint: Hint(stride: [2, 2]))
  var out = maxPool(x)
  
  var layerConv2dArray =  [Convolution]()
  
  for _ in 0..<convLayers {
    let layerConv2d = Convolution(
    groups: 1, filters: outputChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
    out = layerConv2d(out)
    out = ReLU()(out)
    layerConv2dArray.append(layerConv2d)
  }

  return Model([x], [out])
}

func HEDModel(inputWidth: Int, inputHeight: Int) -> Model {
  let x = Input()

  let vggOneInLayerConv2d = Convolution(
    groups: 1, filters: 64, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var vggOneEmbeds = vggOneInLayerConv2d(x)
  vggOneEmbeds = ReLU()(vggOneEmbeds)

  let vggOneOutLayerConv2d = Convolution(
    groups: 1, filters: 64, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  vggOneEmbeds = vggOneOutLayerConv2d(vggOneEmbeds)
  vggOneEmbeds = ReLU()(vggOneEmbeds)

  let vggTwo = vggConvLayer(outputChannels:128, convLayers: 2)
  let vggTwoEmbeds = vggTwo(vggOneEmbeds)
  
  let vggThr = vggConvLayer(outputChannels:256, convLayers: 3)
  let vggThrEmbeds = vggThr(vggTwoEmbeds)

  let vggFour = vggConvLayer(outputChannels:512, convLayers: 3)
  let vggFouEmbeds = vggFour(vggThrEmbeds)

  let vggFiv = vggConvLayer(outputChannels:512, convLayers: 3)
  let vggFiveEmbeds = vggFiv(vggFouEmbeds)

  let netScoreOne = Convolution(
    groups: 1, filters: 1, filterSize: [1, 1],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])))
 
  let netScoreTwo = Convolution(
    groups: 1, filters: 1, filterSize: [1, 1],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])))
 
  let netScoreThr = Convolution(
    groups: 1, filters: 1, filterSize: [1, 1],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])))

    let netScoreFou = Convolution(
    groups: 1, filters: 1, filterSize: [1, 1],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])))

    let netScoreFiv = Convolution(
    groups: 1, filters: 1, filterSize: [1, 1],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])))

   let scaledNetScoreOneEmbeds = Upsample(.bilinear, widthScale: 1, heightScale: 1)(netScoreOne(vggOneEmbeds))
   let scaledNetScoreTwoEmbeds = Upsample(.bilinear, widthScale: 2, heightScale: 2)(netScoreTwo(vggTwoEmbeds))
   let scaledNetScoreThrEmbeds = Upsample(.bilinear, widthScale: 4, heightScale: 4)(netScoreThr(vggThrEmbeds))
   let scaledNetScoreFouEmbeds = Upsample(.bilinear, widthScale: 8, heightScale: 8)(netScoreFou(vggFouEmbeds))
   let scaledNetScoreFivEmbeds = Upsample(.bilinear, widthScale: 16, heightScale: 16)(netScoreFiv(vggFiveEmbeds))
   let mergedVggEmbedings = Functional.concat(axis: 1, scaledNetScoreOneEmbeds, scaledNetScoreTwoEmbeds, scaledNetScoreThrEmbeds, scaledNetScoreFouEmbeds, scaledNetScoreFivEmbeds)

  let netcombineLayerConv2d = Convolution(
    groups: 1, filters: 1, filterSize: [1, 1],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])))
  var out = netcombineLayerConv2d(mergedVggEmbedings)
  out = Sigmoid()(out)

  return  Model([x], [out, vggOneEmbeds])
}


let graph = DynamicGraph()
graph.withNoGrad {
  let tenInput1 = tenInput.type(torch.float).cpu().numpy()
  let tenInputTensor = graph.variable(try! Tensor<Float>(numpy: tenInput1)).toGPU(0)
  let shape = tenInputTensor.shape
  let inputWidth = shape[2]
  let inputHeight = shape[3]

  // print("after tenInputTensor:")
  // debugPrint(tenInput)
  let hedModelVgg = HEDModel(inputWidth:inputWidth, inputHeight:inputHeight)
  hedModelVgg.compile(inputs: tenInputTensor)
  graph.openStore("/home/wlin1/drawThings/swift-diffusion/examples/hed/hed.ckpt") {
    $0.read("hed_vgg", model: hedModelVgg)
  }

  var modeulResult = hedModelVgg(inputs: tenInputTensor)

  var result = modeulResult[0].as(of: Float.self)
  var a2 = modeulResult[1].as(of: Float.self)
  // debugPrint(a2)

  print("swift merged result:")
  debugPrint(result)
  result = result * 255.0
  result = result.reshaped(.NC(inputHeight, inputWidth))
  let p = result.rawValue.toCPU()
    debugPrint(p)

    // PIL.Image.fromarray((tenOutput.clip(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0] * 255.0).astype(numpy.uint8)).save(arguments_strOut)

  let t = Tensor<Float>(from: p ).makeNumpyArray()
  let r = t.astype(numpy.uint8)
  debugPrint(r)
  print(r.shape)
  Image.fromarray(r).save("/home/wlin1/drawThings/swift-diffusion/examples/hed/pytorch-hed/out2.png")
  // Image.fromarray(p.astype(numpy.uint8)).save("/home/wlin1/drawThings/swift-diffusion/examples/hed/pytorch-hed/out2.png")
}


*/