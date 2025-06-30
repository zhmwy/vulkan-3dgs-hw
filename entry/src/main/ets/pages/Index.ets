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
const caseName: string = "food";


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

  aboutToAppear() {
    // 调用自定义的getSyncPixelMap接口，获取pixelMap
    // this.pixelMap = nativeRender.getSyncPixelMap(this.resmgr, "assets/12_data/AVATAR/images_orig/frame_0000.jpg")
    // this.pixelMap = nativeRender.getSyncPixelMap(this.resmgr, "assets/case1/images/15506175713000.jpg")
    this.pixelMap = nativeRender.getSyncPixelMap(this.resmgr, `${caseName}/images/85864565279156.jpg`)
  }

  async train3DGS(inputPath: string) {
    if (this.isTraining) return;
    this.isTraining = true;
    try {
      nativeRender.initTrainProgress(this.resmgr, (progress: number) => {
        this.progressValue = progress;
      });
      await this.readFiles(filesDir + "/" + caseName + "/images");        // 调用 readFiles 读取文件
      hilog.info(0x0000, TAG, `文件列表已获取，沙盒内文件: ${this.pathList}`);
      this.segmentAllImagesInSandbox();
      await nativeRender.train3DGS(this.resmgr, inputPath);
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

        Button(this.isTraining ? 'Training...' : 'Start Training')
          .onClick(() => {
            if (!this.isTraining) {
              this.train3DGS(caseName);
            }
          })
          .margin(10)
          .enabled(!this.isTraining)

        Progress({ value: this.progressValue, total: 100, type: ProgressType.Ring })
          .width(100)
          .height(100)
          .margin(20)

        Image(this.pixelMap)
          .width(240)
          .height(320)
          .onClick(()=>this.switchImage())
      }
      .width('100%')
      .height('100%')
    }
  }

  private async readFiles(path: string) {
    try {
      // 调用 listFile 获取目录下所有文件
      let options: ListFileOptions = {
        recursion: false,  // 不递归子目录
      };
      hilog.info(0x0000, TAG, `正在读取 ${path}中的文件`);

      // 获取文件列表
      let files = await fs.listFile(path, options);
      // // 将所有文件名存入 pathList
      this.pathList = files.filter(file =>
      // (file.endsWith('.jpg') || file.endsWith('.png'))
      (file.endsWith('.jpg'))
      );

    } catch (err) {
      hilog.error(0x0000, TAG, `Error reading files: ${err.message}`);
    }
  }
  private async segmentAllImagesInSandbox() {
    fs.mkdir(getContext().filesDir + `/${caseName}/segmented/`);

    for (let path of this.pathList) {
      let fullPath = getContext().filesDir + `/${caseName}/images/` + path;
      hilog.info(0x0000, TAG, `Segmenting ${fullPath}`);

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
      hilog.info(0x0000, TAG, `图像 ${path} 分割完成，分割数量: ${data.subjectCount}`);

      if (data.fullSubject && data.fullSubject.foregroundImage) {
        // Convert foreground image to binary
        let binaryPixelMap = await this.convertToBinaryImage(data.fullSubject.foregroundImage);

        let fileName = path.split('/').pop() || 'unknown.jpg'
        let savePath = getContext().filesDir + `/${caseName}/segmented/` + fileName;
        hilog.info(0x0000, TAG, `目标路径: ${savePath}`);
        let file = fs.openSync(savePath, fs.OpenMode.CREATE | fs.OpenMode.READ_WRITE);

        // Save as PNG to preserve the binary nature without compression artifacts
        await image.createImagePacker().packToFile(binaryPixelMap, file.fd, {
          format: "image/png",
          quality: 100
        });
        fs.closeSync(file);
        hilog.info(0x0000, TAG, `保存分割图: ${savePath}`);
      } else {
        hilog.warn(0x0000, TAG, `图像 ${path} 分割无前景图`);
      }
    }
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