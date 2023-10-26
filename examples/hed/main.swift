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
let imagePath = "/home/wlin1/drawThings/swift-diffusion/examples/hed/pytorch-hed/images/sample.png"

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
//   hedVggReader(state_dict)
  graph.openStore("/home/wlin1/drawThings/swift-diffusion/examples/hed/hed.ckpt") {
    $0.read("hed_vgg", model: hedModelVgg)
  }

  let modeulResult = hedModelVgg(inputs: tenInputTensor)[0].as(of: Float.self)

//    graph.openStore("/home/wlin1/drawThings/swift-diffusion/examples/hed/hed.ckpt") {
//     $0.write("hed_vgg", model: hedModelVgg)
//   }
  print("swift merged result:")
  debugPrint(modeulResult)
}