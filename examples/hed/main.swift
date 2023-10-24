import Diffusion
import Foundation
import NNC
import NNCPythonConversion
import PNG
import PythonKit


print("hello world")
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


func netVggOne() -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let inLayerConv2d = Convolution(
    groups: 1, filters: 64, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var out = inLayerConv2d(x)
  out = ReLU()(out)

  let outLayerConv2d = Convolution(
    groups: 1, filters: 64, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = outLayerConv2d(out)
  out = ReLU()(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let inLayerConv2d_weight = state_dict["netVggOne.0.weight"].type(torch.float).cpu().numpy()
    let inLayerConv2d_bias = state_dict["netVggOne.0.bias"].type(torch.float).cpu().numpy()
    inLayerConv2d.weight.copy(from: try! Tensor<Float>(numpy: inLayerConv2d_weight))
    inLayerConv2d.bias.copy(from: try! Tensor<Float>(numpy: inLayerConv2d_bias))
    let outLayerConv2d_weight = state_dict["netVggOne.2.weight"].type(torch.float).cpu().numpy()
    let outLayerConv2d_bias = state_dict["netVggOne.2.bias"].type(torch.float).cpu().numpy()
    outLayerConv2d.weight.copy(from: try! Tensor<Float>(numpy: outLayerConv2d_weight))
    outLayerConv2d.bias.copy(from: try! Tensor<Float>(numpy: outLayerConv2d_bias))
  }

  return (reader, Model([x], [out]))
}

func netVggTwo() -> ((PythonObject) -> Void, Model) {
  let x = Input()

  let maxPool = MaxPool(filterSize: [2, 2], hint: Hint(stride: [2, 2]))
  var out = maxPool(x)

  let inLayerConv2d = Convolution(
    groups: 1, filters: 128, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = inLayerConv2d(out)
  out = ReLU()(out)

  let outLayerConv2d = Convolution(
    groups: 1, filters: 128, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = outLayerConv2d(out)
  out = ReLU()(out)

  let reader: (PythonObject) -> Void = { state_dict in
    let inLayerConv2d_weight = state_dict["netVggTwo.1.weight"].type(torch.float).cpu().numpy()
    let inLayerConv2d_bias = state_dict["netVggTwo.1.bias"].type(torch.float).cpu().numpy()
    inLayerConv2d.weight.copy(from: try! Tensor<Float>(numpy: inLayerConv2d_weight))
    inLayerConv2d.bias.copy(from: try! Tensor<Float>(numpy: inLayerConv2d_bias))

    let outLayerConv2d_weight = state_dict["netVggTwo.3.weight"].type(torch.float).cpu().numpy()
    let outLayerConv2d_bias = state_dict["netVggTwo.3.bias"].type(torch.float).cpu().numpy()
    outLayerConv2d.weight.copy(from: try! Tensor<Float>(numpy: outLayerConv2d_weight))
    outLayerConv2d.bias.copy(from: try! Tensor<Float>(numpy: outLayerConv2d_bias))
  }

  return (reader, Model([x], [out]))
}

func netVggThree() -> ((PythonObject) -> Void, Model) {
  let x = Input()

  let maxPool = MaxPool(filterSize: [2, 2], hint: Hint(stride: [2, 2]))
  var out = maxPool(x)

  let inLayerConv2d = Convolution(
    groups: 1, filters: 256, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = inLayerConv2d(out)
  out = ReLU()(out)

  let midLayerConv2d = Convolution(
    groups: 1, filters: 256, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = midLayerConv2d(out)
  out = ReLU()(out)

  let outLayerConv2d = Convolution(
    groups: 1, filters: 256, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = outLayerConv2d(out)
  out = ReLU()(out)

  let reader: (PythonObject) -> Void = { state_dict in
    let inLayerConv2d_weight = state_dict["netVggThr.1.weight"].type(torch.float).cpu().numpy()
    let inLayerConv2d_bias = state_dict["netVggThr.1.bias"].type(torch.float).cpu().numpy()
    inLayerConv2d.weight.copy(from: try! Tensor<Float>(numpy: inLayerConv2d_weight))
    inLayerConv2d.bias.copy(from: try! Tensor<Float>(numpy: inLayerConv2d_bias))

    let midLayerConv2d_weight = state_dict["netVggThr.3.weight"].type(torch.float).cpu().numpy()
    let midLayerConv2d_bias = state_dict["netVggThr.3.bias"].type(torch.float).cpu().numpy()
    midLayerConv2d.weight.copy(from: try! Tensor<Float>(numpy: midLayerConv2d_weight))
    midLayerConv2d.bias.copy(from: try! Tensor<Float>(numpy: midLayerConv2d_bias))


    let outLayerConv2d_weight = state_dict["netVggThr.5.weight"].type(torch.float).cpu().numpy()
    let outLayerConv2d_bias = state_dict["netVggThr.5.bias"].type(torch.float).cpu().numpy()
    outLayerConv2d.weight.copy(from: try! Tensor<Float>(numpy: outLayerConv2d_weight))
    outLayerConv2d.bias.copy(from: try! Tensor<Float>(numpy: outLayerConv2d_bias))
  }

  return (reader, Model([x], [out]))
}

func netVggFour() -> ((PythonObject) -> Void, Model) {
  let x = Input()

  let maxPool = MaxPool(filterSize: [2, 2], hint: Hint(stride: [2, 2]))
  var out = maxPool(x)

  let inLayerConv2d = Convolution(
    groups: 1, filters: 512, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = inLayerConv2d(out)
  out = ReLU()(out)

  let midLayerConv2d = Convolution(
    groups: 1, filters: 512, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = midLayerConv2d(out)
  out = ReLU()(out)

  let outLayerConv2d = Convolution(
    groups: 1, filters: 512, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = outLayerConv2d(out)
  out = ReLU()(out)

  let reader: (PythonObject) -> Void = { state_dict in
    let inLayerConv2d_weight = state_dict["netVggFou.1.weight"].type(torch.float).cpu().numpy()
    let inLayerConv2d_bias = state_dict["netVggFou.1.bias"].type(torch.float).cpu().numpy()
    inLayerConv2d.weight.copy(from: try! Tensor<Float>(numpy: inLayerConv2d_weight))
    inLayerConv2d.bias.copy(from: try! Tensor<Float>(numpy: inLayerConv2d_bias))

    let midLayerConv2d_weight = state_dict["netVggFou.3.weight"].type(torch.float).cpu().numpy()
    let midLayerConv2d_bias = state_dict["netVggFou.3.bias"].type(torch.float).cpu().numpy()
    midLayerConv2d.weight.copy(from: try! Tensor<Float>(numpy: midLayerConv2d_weight))
    midLayerConv2d.bias.copy(from: try! Tensor<Float>(numpy: midLayerConv2d_bias))


    let outLayerConv2d_weight = state_dict["netVggFou.5.weight"].type(torch.float).cpu().numpy()
    let outLayerConv2d_bias = state_dict["netVggFou.5.bias"].type(torch.float).cpu().numpy()
    outLayerConv2d.weight.copy(from: try! Tensor<Float>(numpy: outLayerConv2d_weight))
    outLayerConv2d.bias.copy(from: try! Tensor<Float>(numpy: outLayerConv2d_bias))
  }

  return (reader, Model([x], [out]))
}

func netVggFive() -> ((PythonObject) -> Void, Model) {
  let x = Input()

  let maxPool = MaxPool(filterSize: [2, 2], hint: Hint(stride: [2, 2]))
  var out = maxPool(x)

  let inLayerConv2d = Convolution(
    groups: 1, filters: 512, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = inLayerConv2d(out)
  out = ReLU()(out)

  let midLayerConv2d = Convolution(
    groups: 1, filters: 512, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = midLayerConv2d(out)
  out = ReLU()(out)

  let outLayerConv2d = Convolution(
    groups: 1, filters: 512, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = outLayerConv2d(out)
  out = ReLU()(out)

  let reader: (PythonObject) -> Void = { state_dict in
    let inLayerConv2d_weight = state_dict["netVggFiv.1.weight"].type(torch.float).cpu().numpy()
    let inLayerConv2d_bias = state_dict["netVggFiv.1.bias"].type(torch.float).cpu().numpy()
    inLayerConv2d.weight.copy(from: try! Tensor<Float>(numpy: inLayerConv2d_weight))
    inLayerConv2d.bias.copy(from: try! Tensor<Float>(numpy: inLayerConv2d_bias))

    let midLayerConv2d_weight = state_dict["netVggFiv.3.weight"].type(torch.float).cpu().numpy()
    let midLayerConv2d_bias = state_dict["netVggFiv.3.bias"].type(torch.float).cpu().numpy()
    midLayerConv2d.weight.copy(from: try! Tensor<Float>(numpy: midLayerConv2d_weight))
    midLayerConv2d.bias.copy(from: try! Tensor<Float>(numpy: midLayerConv2d_bias))


    let outLayerConv2d_weight = state_dict["netVggFiv.5.weight"].type(torch.float).cpu().numpy()
    let outLayerConv2d_bias = state_dict["netVggFiv.5.bias"].type(torch.float).cpu().numpy()
    outLayerConv2d.weight.copy(from: try! Tensor<Float>(numpy: outLayerConv2d_weight))
    outLayerConv2d.bias.copy(from: try! Tensor<Float>(numpy: outLayerConv2d_bias))
  }

  return (reader, Model([x], [out]))
}

func netCombine() -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let inLayerConv2d = Convolution(
    groups: 1, filters: 1, filterSize: [1, 1],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])))
  var out = inLayerConv2d(x)
  out = Sigmoid()(out)
  
  let reader: (PythonObject) -> Void = { state_dict in
    let inLayerConv2d_weight = state_dict["netCombine.0.weight"].type(torch.float).cpu().numpy()
    let inLayerConv2d_bias = state_dict["netCombine.0.bias"].type(torch.float).cpu().numpy()
    inLayerConv2d.weight.copy(from: try! Tensor<Float>(numpy: inLayerConv2d_weight))
    inLayerConv2d.bias.copy(from: try! Tensor<Float>(numpy: inLayerConv2d_bias))
  }

  return (reader, Model([x], [out]))
}

func HedModelVggEmbeds(inputWidth: Int, inputHeight: Int) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let (reader, vggOne) = netVggOne()
  let vggOneEmbeds = vggOne(x)

  let (reader2, vggTwo) = netVggTwo()
  let vggTwoEmbeds = vggTwo(vggOneEmbeds)

  let (reader3, vggThr) = netVggThree()
  let vggThrEmbeds = vggThr(vggTwoEmbeds)

  let (reader4, vggFour) = netVggFour()
  let vggFouEmbeds = vggFour(vggThrEmbeds)

  let (reader5, vggFiv) = netVggFive()
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

   let netScoreOneEmbeds = netScoreOne(vggOneEmbeds)
   let netScoreTwoEmbeds = netScoreTwo(vggTwoEmbeds)
   let netScoreThrEmbeds = netScoreThr(vggThrEmbeds)
   let netScoreFouEmbeds = netScoreFou(vggFouEmbeds)
   let netScoreFivEmbeds = netScoreFiv(vggFiveEmbeds)

  let readerExternal: (PythonObject) -> Void = { state_dict in
    reader(state_dict)
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

  }

  return (readerExternal, Model([x], [netScoreOneEmbeds, netScoreTwoEmbeds, netScoreThrEmbeds, netScoreFouEmbeds, netScoreFivEmbeds]))
}

func HedModelMergeVggEmbedings( vggEmbedings : [DynamicGraph.Tensor<Float32>], inputWidth: Int, inputHeight: Int) -> DynamicGraph.Tensor<Float32> {
    precondition(vggEmbedings.count  == 5)

   let netScoreOneEmbeds = vggEmbedings[0]
   let netScoreTwoEmbeds = vggEmbedings[1]
   let netScoreThrEmbeds = vggEmbedings[2]
   let netScoreFouEmbeds = vggEmbedings[3]
   let netScoreFivEmbeds = vggEmbedings[4]
   
   let n1_width = netScoreOneEmbeds.shape[2]
   let n1_height = netScoreOneEmbeds.shape[3]
   let scaledNetScoreOneEmbeds = Upsample(.bilinear, widthScale: Float(inputWidth)/Float(n1_width), heightScale: Float(inputHeight)/Float(n1_height))(netScoreOneEmbeds)

   let n2_width = netScoreTwoEmbeds.shape[2]
   let n2_height = netScoreTwoEmbeds.shape[3]
   let scaledNetScoreTwoEmbeds = Upsample(.bilinear, widthScale: Float(inputWidth)/Float(n2_width), heightScale: Float(inputHeight)/Float(n2_height))(netScoreTwoEmbeds)

   let n3_width = netScoreThrEmbeds.shape[2]
   let n3_height = netScoreThrEmbeds.shape[3]
   let scaledNetScoreThrEmbeds = Upsample(.bilinear, widthScale: Float(inputWidth)/Float(n3_width), heightScale: Float(inputHeight)/Float(n3_height))(netScoreThrEmbeds)

   let n4_width = netScoreFouEmbeds.shape[2]
   let n4_height = netScoreFouEmbeds.shape[3]
   let scaledNetScoreFouEmbeds = Upsample(.bilinear, widthScale: Float(inputWidth)/Float(n4_width), heightScale: Float(inputHeight)/Float(n4_height))(netScoreFouEmbeds)

   let n5_width = netScoreFivEmbeds.shape[2]
   let n5_height = netScoreFivEmbeds.shape[3]
   let scaledNetScoreFivEmbeds = Upsample(.bilinear, widthScale: Float(inputWidth)/Float(n5_width), heightScale: Float(inputHeight)/Float(n5_height))(netScoreFivEmbeds)

   var mergedVggEmbedings = graph.variable(.GPU(0), .NCHW(1, 5, inputWidth, inputHeight), of: Float.self)

   mergedVggEmbedings[0..<1, 0..<1, 0..<inputWidth, 0..<inputHeight] = scaledNetScoreOneEmbeds
   mergedVggEmbedings[0..<1, 1..<2, 0..<inputWidth, 0..<inputHeight] = scaledNetScoreTwoEmbeds
   mergedVggEmbedings[0..<1, 2..<3, 0..<inputWidth, 0..<inputHeight] = scaledNetScoreThrEmbeds
   mergedVggEmbedings[0..<1, 3..<4, 0..<inputWidth, 0..<inputHeight] = scaledNetScoreFouEmbeds
   mergedVggEmbedings[0..<1, 4..<5, 0..<inputWidth, 0..<inputHeight] = scaledNetScoreFivEmbeds

   return mergedVggEmbedings
}

func HedModelNetCombine() -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let inLayerConv2d = Convolution(
    groups: 1, filters: 1, filterSize: [1, 1],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])))
  var out = inLayerConv2d(x)
  out = Sigmoid()(out)
  
  let reader: (PythonObject) -> Void = { state_dict in
    let inLayerConv2d_weight = state_dict["netCombine.0.weight"].type(torch.float).cpu().numpy()
    let inLayerConv2d_bias = state_dict["netCombine.0.bias"].type(torch.float).cpu().numpy()
    inLayerConv2d.weight.copy(from: try! Tensor<Float>(numpy: inLayerConv2d_weight))
    inLayerConv2d.bias.copy(from: try! Tensor<Float>(numpy: inLayerConv2d_bias))
  }

  return (reader, Model([x], [out]))
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

  let hedModelEmbeds = hedModelVgg(inputs: tenInputTensor).map {  $0.as(of: Float.self) }
  let mergedVggEmbedings = HedModelMergeVggEmbedings(vggEmbedings:hedModelEmbeds, inputWidth:inputWidth, inputHeight:inputHeight)
  let (readerNetCombine, vggNetCombine) = netCombine()

   vggNetCombine.compile(inputs: mergedVggEmbedings)
//    readerNetCombine(state_dict)
  graph.openStore("/home/wlin1/drawThings/swift-diffusion/examples/hed/hed.ckpt") {
    $0.read("vgg_combine", model: vggNetCombine)
  }
   let modeulResult = vggNetCombine(inputs:mergedVggEmbedings)

//    graph.openStore("/home/wlin1/drawThings/swift-diffusion/examples/hed/hed.ckpt") {
//     $0.write("hed_vgg", model: hedModelVgg)
//     $0.write("vgg_combine", model: vggNetCombine)
//   }
  print("swift merged result:")
  debugPrint(modeulResult)
}