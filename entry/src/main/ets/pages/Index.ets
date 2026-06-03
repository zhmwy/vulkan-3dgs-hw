import { hilog } from '@kit.PerformanceAnalysisKit';
// import nativeRender from 'lib3dgs.so';
import nativeRender from 'libinput.so';
import CommonConstants from '../common/CommonConstants';
// import image from '@ohos.multimedia.image'
import { subjectSegmentation } from '@kit.CoreVisionKit';
import { image } from '@kit.ImageKit';
import { BusinessError } from '@kit.BasicServicesKit';
import { fileIo, ListFileOptions } from '@kit.CoreFileKit';
import { photoAccessHelper } from '@kit.MediaLibraryKit';
import { fileIo as fs } from '@kit.CoreFileKit';  // 导入 fs (fileIo)

const context = getContext(this);
const filesDir = context.filesDir;
const TAG: string = "ImageSegmentationSample";

const caseNames: string[] = [
  "2025-06-04_15_34_45",
  "2025-06-04_15_57_44",
  "2025-06-04_15_59_26",
  "2025-06-04_19_28_34",
  "2025-06-04_19_29_27",
  "2025-06-04_19_30_49",
  "2025-06-04_19_31_45",
  "2025-06-04_19_32_45",
  "2025-06-04_19_35_58",
  "2025-06-04_19_40_24",
  "2025-06-04_19_43_28",
  "2025-06-04_19_45_20",
  "2025-06-04_19_46_14",
  "2025-06-04_19_49_06",
  "2025-06-04_20_09_08",
  "2025-06-04_20_10_08",
  "2025-06-04_20_12_31",
  "2025-06-04_20_13_31",
  "2025-06-04_20_14_32",
  "2025-06-04_20_15_40",
  "2025-06-04_20_22_53",
  "2025-06-04_20_26_45",
  "2025-06-05_20_27_20",
  "2025-06-05_20_34_42",
  "2025-06-05_20_38_24"
]

const chosenCase: string[] = [
  "2025-06-04_15_34_45",
  "2025-06-04_19_40_24",
  "2025-06-04_20_26_45",
  "2025-06-05_20_27_20"
]

const sceneCases: string[] = [
  "01", "02", "03", "04", "05",
  "06", "07", "08", "09", "10",
  "11", "12", "13", "14", "15",
  "16", "17", "18", "19", "20",
]


let caseName = caseNames[10];

class GS_Mode {
  static readonly SCENE = 0 // 暂无特定优化
  static readonly OBJECT = 1 //
  static readonly HUMAN = 2 // SMPL估计、人脸重点、美颜
  static readonly MULTI_HUMAN = 3 // 人脸重点
}
const trainingMode: number = GS_Mode.OBJECT;

const isObject: boolean = false;

@Entry
@Component
struct Index {
  @State chooseImage: PixelMap | undefined = undefined
  @State dataValues: string = ''
  @State segmentedImage: PixelMap | undefined = undefined
  @State maxNum: string = '20'
  @State selectedImagePath: string | undefined = undefined; // 新增
  @State pathList: string[] = []

  private xComponentContext: Record<string, (a: number | boolean | undefined) => void> = {};
  private resmgr = getContext().resourceManager;
  @State progressValue: number = 0;
  @State message: string = "3DGS Training";
  @State isTraining: boolean = false;
  @State pixelMap : PixelMap | undefined = undefined;
  @State currentImage: number = 0;
  @State segmentViews: number = 0;

  aboutToAppear() {
    // 调用自定义的getSyncPixelMap接口，获取pixelMap
    // this.pixelMap = nativeRender.getSyncPixelMap(this.resmgr, "assets/12_data/AVATAR/images_orig/frame_0000.jpg")
    // this.pixelMap = nativeRender.getSyncPixelMap(this.resmgr, "assets/case1/images/15506175713000.jpg")
    this.pixelMap = nativeRender.getSyncPixelMap(this.resmgr, `${caseName}/images/85864565279156.jpg`)
    this.train3DGS(caseName);
  }

  async train3DGS(inputPath: string) {
    if (this.isTraining) return;
    this.isTraining = true;
    try {
      nativeRender.initTrainProgress(this.resmgr, (progress: number) => {
        this.progressValue = progress;
      });
      // for (let i = 4; i < caseNames.length; i++){
      //   caseName = caseNames[i];
      //   await this.readFiles(filesDir + "/" + caseName + "/images");        // 调用 readFiles 读取文件
      //   hilog.info(0x0000, TAG, `文件列表已获取，沙盒内文件: ${this.pathList}`);
      //   await this.segmentAllImagesInSandbox();
      //   // await nativeRender.train3DGS(this.resmgr, caseName);  // for循环
      //   this.message = "Training Complete!";
      // }
      // caseName = caseNames[4];
      await this.readFiles(filesDir + "/" + caseName + "/images", this.segmentViews);        // 调用 readFiles 读取文件kl;lkm
      // hilog.info(0x0000, TAG, `文件列表已获取，沙盒内文件: ${this.pathList}`)

      if(trainingMode == GS_Mode.OBJECT){
        this.segmentAllImagesInSandbox();
      }

      await nativeRender.train3DGS(this.resmgr, caseName, GS_Mode.OBJECT, isObject);  // for循环
      this.message = "Training Complete!";
    } catch (error) {
      // Handle error
    }
  }

  async switchImage(){

    this.pixelMap = nativeRender.getSandBoxPixelMap(filesDir, "/" + caseName + "/temp" + this.currentImage.toString() + ".jpg")
    this.currentImage = (this.currentImage + 1) % 1000
  }



  build() {
    Row() {
      Column() {
        Text(this.message)
          .fontSize(50)
          .fontWeight(FontWeight.Bold)
          .margin(10)

        // Button(this.isTraining ? 'Training...' : 'Start Training')
        //   .onClick(() => {
        //     if (!this.isTraining) {
        //       this.train3DGS(caseName);
        //     }
        //   })
        //   .margin(10)
        //   .enabled(!this.isTraining)

        Progress({ value: this.progressValue, total: 100, type: ProgressType.Ring })
          .width(100)
          .height(100)
          .margin(20)

        // Image(this.pixelMap)
        //   .width(240)
        //   .height(320)
        //   .onClick(()=>this.switchImage())
      }
      .width('100%')
      .height('100%')
    }
  }

  private async readFiles(path: string, numToSelect?: number) {
    try {
      // 调用 listFile 获取目录下所有文件
      let options: ListFileOptions = {
        recursion: false,  // 不递归子目录
      };
      // hilog.info(0x0000, TAG, `正在读取 ${path}中的文件`);

      // 获取文件列表
      let files = await fs.listFile(path, options);

      // 过滤出所有图片文件
      let allImages = files.filter(file =>
      (file.endsWith('.jpg') || file.endsWith('.png'))
      );

      // 如果指定了选择数量，则均匀选择N张图片
      if (numToSelect && numToSelect > 0 && numToSelect < allImages.length) {
        this.pathList = this.selectImagesEvenly(allImages, numToSelect);
        hilog.info(0x0000, TAG, `从 ${allImages.length} 张图片中均匀选择了 ${this.pathList.length} 张`);
      } else {
        // 否则使用所有图片
        this.pathList = allImages;
        // hilog.info(0x0000, TAG, `使用所有 ${this.pathList.length} 张图片`);
      }

    } catch (err) {
      hilog.error(0x0000, TAG, `Error reading files: ${err.message}`);
    }
  }

  // 新增：均匀选择N张图片的方法
  private selectImagesEvenly(images: string[], count: number): string[] {
    if (count >= images.length) {
      return images;
    }

    let selected: string[] = [];
    let totalImages = images.length;
    let step = (totalImages - 1) / (count - 1); // 计算步长

    for (let i = 0; i < count; i++) {
      let index = Math.round(i * step);
      // 确保索引不超出范围
      index = Math.min(index, totalImages - 1);
      selected.push(images[index]);
    }

    return selected;
  }

  private async segmentAllImagesInSandbox() {
    fs.mkdir(getContext().filesDir + `/${caseName}/segmented/`);

    // hilog.info(0x0000, TAG, `开始分割 ${this.pathList.length} 张图片`);

    for (let i = 0; i < this.pathList.length; i++) {
      let path = this.pathList[i];
      let fullPath = getContext().filesDir + `/${caseName}/images/` + path;
      // hilog.info(0x0000, TAG, `正在分割第 ${i + 1}/${this.pathList.length} 张: ${path}`);

      let fileSource = await fileIo.open(fullPath, fileIo.OpenMode.READ_ONLY)

      let imageSource = image.createImageSource(fileSource.fd)
      let pixelMap = await imageSource.createPixelMap()

      let visionInfo: subjectSegmentation.VisionInfo = {
        pixelMap: pixelMap,
      };
      let config: subjectSegmentation.SegmentationConfig = {
        maxCount: parseInt(this.maxNum),
        enableSubjectDetails: true,
        enableSubjectForegroundImage: true,
      };

      let data = await subjectSegmentation.doSegmentation(visionInfo, config);

      if (data.fullSubject && data.fullSubject.foregroundImage) {
        // Convert foreground image to binary
        let binaryPixelMap = await this.convertToBinaryImage(data.fullSubject.foregroundImage);

        let fileName = path.split('/').pop();
        fileName = fileName?.replace('.jpg', '.png').replace('.png', '.png'); // 确保都是.png
        let savePath = getContext().filesDir + `/${caseName}/segmented/` + fileName;

        let file = fs.openSync(savePath, fs.OpenMode.CREATE | fs.OpenMode.READ_WRITE);

        // Save as PNG to preserve the binary nature without compression artifacts
        await image.createImagePacker().packToFile(binaryPixelMap, file.fd, {
          format: "image/png",
          quality: 100
        });
        fs.closeSync(file);
        // hilog.info(0x0000, TAG, `已保存分割图 ${i + 1}/${this.pathList.length}: ${savePath}`);
      } else {
        hilog.warn(0x0000, TAG, `图像 ${path} 分割无前景图`);
      }

      // 释放资源
      fileIo.close(fileSource);
    }

    hilog.info(0x0000, TAG, `所有图片分割完成，共处理 ${this.pathList.length} 张`);
  }

  private async convertToBinaryImage(sourcePixelMap: image.PixelMap): Promise<image.PixelMap> {
    // Get image info
    let imageInfo = await sourcePixelMap.getImageInfo();
    let width = imageInfo.size.width;
    let height = imageInfo.size.height;

    // Read pixel data
    let bufferSize = width * height * 4; // RGBA format
    let buffer = new ArrayBuffer(bufferSize);
    await sourcePixelMap.readPixelsToBuffer(buffer);

    // Convert to Uint8Array for easier manipulation
    let pixels = new Uint8Array(buffer);

    // Create binary pixel data
    let binaryBuffer = new ArrayBuffer(bufferSize);
    let binaryPixels = new Uint8Array(binaryBuffer);

    // Process each pixel
    for (let i = 0; i < pixels.length; i += 4) {
      // Check alpha channel to determine if pixel is foreground
      let alpha = pixels[i + 3];

      if (alpha > 0) { // Foreground (subject)
        // Set to white
        binaryPixels[i] = 255;     // R
        binaryPixels[i + 1] = 255; // G
        binaryPixels[i + 2] = 255; // B
        binaryPixels[i + 3] = 255; // A
      } else { // Background
        // Set to black
        binaryPixels[i] = 0;       // R
        binaryPixels[i + 1] = 0;   // G
        binaryPixels[i + 2] = 0;   // B
        binaryPixels[i + 3] = 255; // A (fully opaque)
      }
    }

    // Create new PixelMap with binary data
    let opts: image.InitializationOptions = {
      size: {
        width: width,
        height: height
      },
      pixelFormat: image.PixelMapFormat.RGBA_8888,
      editable: true
    };

    let binaryPixelMap = await image.createPixelMap(binaryBuffer, opts);
    return binaryPixelMap;
  }
}