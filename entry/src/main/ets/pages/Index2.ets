import { subjectSegmentation } from '@kit.CoreVisionKit';
import { image } from '@kit.ImageKit';
import { hilog } from '@kit.PerformanceAnalysisKit';
import { BusinessError } from '@kit.BasicServicesKit';
import { fileIo, ListFileOptions } from '@kit.CoreFileKit';
import { photoAccessHelper } from '@kit.MediaLibraryKit';
import { fileIo as fs } from '@kit.CoreFileKit';  // 导入 fs (fileIo)


const TAG: string = "ImageSegmentationSample";

@Entry
@Component
struct Index {
  @State chooseImage: PixelMap | undefined = undefined
  @State dataValues: string = ''
  @State segmentedImage: PixelMap | undefined = undefined
  @State maxNum: string = '20'
  @State selectedImagePath: string | undefined = undefined; // 新增
  @State pathList: string[] = []

  build() {
    Column() {
      Image(this.chooseImage)
        .objectFit(ImageFit.Fill)
        .height('30%')
        .accessibilityDescription("Image to be segmented")

      Scroll() {
        Text(this.dataValues)
          .copyOption(CopyOptions.LocalDevice)
          .margin(10)
          .width('100%')
      }
      .height('20%')

      Image(this.segmentedImage)
        .objectFit(ImageFit.Fill)
        .height('30%')
        .accessibilityDescription("Segmented subject image")

      Row() {
        Text('Max subject count:')
          .fontSize(16)
        TextInput({ placeholder: 'Enter max subject count', text: this.maxNum })
          .type(InputType.Number)
          .placeholderColor(Color.Gray)
          .fontSize(16)
          .backgroundColor(Color.White)
          .onChange((value: string) => {
            this.maxNum = value
          })
      }
      .width('80%')
      .margin(10)

      Button('Select Image')
        .type(ButtonType.Capsule)
        .fontColor(Color.White)
        .alignSelf(ItemAlign.Center)
        .width('80%')
        .margin(10)
        .onClick(() => {
          this.selectImage()
        })
      Button('Sandbox')
        .type(ButtonType.Capsule)
        .fontColor(Color.White)
        .alignSelf(ItemAlign.Center)
        .width('80%')
        .margin(10)
        .onClick(async() => {
          let path = getContext().filesDir;  // 获取文件夹路径
          await this.readFiles(path + "/human_bie/images/");        // 调用 readFiles 读取文件
          hilog.info(0x0000, TAG, `正在读取 ${path + "/human_bie/images/"}`);
          hilog.info(0x0000, TAG, `文件列表已获取，沙盒内文件: ${this.pathList}`);
        })
      Button('Batch Segment')
        .type(ButtonType.Capsule)
        .fontColor(Color.White)
        .alignSelf(ItemAlign.Center)
        .width('80%')
        .margin(10)
        .onClick(async () => {
          await this.segmentAllImagesInSandbox()
        })

      Button('Image Segmentation')
        .type(ButtonType.Capsule)
        .fontColor(Color.White)
        .alignSelf(ItemAlign.Center)
        .width('80%')
        .margin(10)
        .onClick(() => {
          if (!this.chooseImage) {
            hilog.error(0x0000, TAG, "imageSegmentation not have chooseImage");
            return
          }
          let visionInfo: subjectSegmentation.VisionInfo = {
            pixelMap: this.chooseImage,
          };
          let config: subjectSegmentation.SegmentationConfig = {
            maxCount: parseInt(this.maxNum),
            enableSubjectDetails: true,
            enableSubjectForegroundImage: true,
          };
          subjectSegmentation.doSegmentation(visionInfo, config)
            .then((data: subjectSegmentation.SegmentationResult) => {
              let outputString = `Subject count: ${data.subjectCount}\n`;
              outputString += `Max subject count: ${config.maxCount}\n`;
              outputString += `Enable subject details: ${config.enableSubjectDetails ? 'Yes' : 'No'}\n\n`;
              let segBox : subjectSegmentation.Rectangle = data.fullSubject.subjectRectangle;
              let segBoxString = `Full subject box:\nLeft: ${segBox.left}, Top: ${segBox.top}, Width: ${segBox.width}, Height: ${segBox.height}\n\n`;
              outputString += segBoxString;

              if (config.enableSubjectDetails) {
                outputString += 'Individual subject boxes:\n';
                if (data.subjectDetails) {
                  for (let i = 0; i < data.subjectDetails.length; i++) {
                    let detailSegBox: subjectSegmentation.Rectangle = data.subjectDetails[i].subjectRectangle;
                    outputString += `Subject ${i + 1}:\nLeft: ${detailSegBox.left}, Top: ${detailSegBox.top}, Width: ${detailSegBox.width}, Height: ${detailSegBox.height}\n\n`;
                  }
                }
              }

              hilog.info(0x0000, TAG, "Segmentation result: " + outputString);
              this.dataValues = outputString;

              if (data.fullSubject && data.fullSubject.foregroundImage) {
                this.segmentedImage = data.fullSubject.foregroundImage;
                // 保存分割后的前景图片
                let tempPath = getContext().filesDir + "/segmented_image.jpg";
                this.saveSegmentedImage(tempPath);
              } else {
                hilog.warn(0x0000, TAG, "No foreground image in segmentation result");
              }
            })
            .catch((error: BusinessError) => {
              hilog.error(0x0000, TAG, `Image segmentation failed errCode: ${error.code}, errMessage: ${error.message}`);
              this.dataValues = `Error: ${error.message}`;
              this.segmentedImage = undefined;
            });
        })
    }
    .width('100%')
    .height('80%')
    .justifyContent(FlexAlign.Center)
  }

  private async selectImage() {
    let uri = await this.openPhoto()
    if (uri === undefined) {
      hilog.error(0x0000, TAG, "uri is undefined");
      return;
    }
    this.selectedImagePath = uri[0]; // 保存路径
    // hilog.info(0x0000, TAG, `uri =  ${uri}`);
    // hilog.info(0x0000, TAG, `uri[0] =  ${uri[0]}`);

    this.loadImage(uri);
    let originalImagePath = getContext().filesDir + "/original_image_" + new Date().getTime() + ".jpg";
    this.saveOriginImage(originalImagePath);
  }


  private openPhoto(): Promise<Array<string>> {
    return new Promise<Array<string>>((resolve, reject) => {
      let PhotoSelectOptions = new photoAccessHelper.PhotoSelectOptions();
      PhotoSelectOptions.MIMEType = photoAccessHelper.PhotoViewMIMETypes.IMAGE_TYPE;
      PhotoSelectOptions.maxSelectNumber = 1;
      let photoPicker: photoAccessHelper.PhotoViewPicker = new photoAccessHelper.PhotoViewPicker();
      hilog.info(0x0000, TAG, 'PhotoViewPicker.select successfully, PhotoSelectResult uri: ');
      photoPicker.select(PhotoSelectOptions).then((PhotoSelectResult) => {
        hilog.info(0x0000, TAG, `PhotoViewPicker.select successfully, PhotoSelectResult uri: ${PhotoSelectResult.photoUris}`);
        resolve(PhotoSelectResult.photoUris)
      }).catch((err: BusinessError) => {
        hilog.error(0x0000, TAG, `PhotoViewPicker.select failed with errCode: ${err.code}, errMessage: ${err.message}`);
        reject();
      });
    })
  }

  private loadImage(names: string[]) {
    setTimeout(async () => {
      let imageSource: image.ImageSource | undefined = undefined
      let fileSource = await fileIo.open(names[0], fileIo.OpenMode.READ_ONLY)
      imageSource = image.createImageSource(fileSource.fd)
      this.chooseImage = await imageSource.createPixelMap()
      hilog.info(0x0000, TAG, `this.chooseImage===${this.chooseImage}`);
    }, 100)
  }

  // 新增的保存方法
  private saveSegmentedImage(savePath: string) {
    if (this.segmentedImage) {
      let file = fs.openSync(savePath, fs.OpenMode.CREATE | fs.OpenMode.READ_WRITE);
      image.createImagePacker().packToFile(this.segmentedImage, file.fd, {
        format: "image/jpeg",
        quality: 100
      }).then(() => {
        fs.closeSync(file);
        console.log("图像保存成功：" + savePath);
      }).catch((err: BusinessError) => {
        fs.closeSync(file);
        console.log("图像保存失败：" + JSON.stringify(err));
      });
    }
  }
  private saveOriginImage(savePath: string) {
    if (this.chooseImage ) {
      let file = fs.openSync(savePath, fs.OpenMode.CREATE | fs.OpenMode.READ_WRITE);
      image.createImagePacker().packToFile(this.chooseImage, file.fd, {
        format: "image/jpeg",
        quality: 100
      }).then(() => {
        fs.closeSync(file);
        console.log("图像保存成功：" + savePath);
      }).catch((err: BusinessError) => {
        fs.closeSync(file);
        console.log("图像保存失败：" + JSON.stringify(err));
      });
    }
  }
  private async readFiles(path: string) {
    try {
      // 调用 listFile 获取目录下所有文件
      let options: ListFileOptions = {
        recursion: false,  // 不递归子目录
      };

      // 获取文件列表
      let files = await fs.listFile(path, options);
      hilog.info(0x0000, TAG, `Files: ${files}`);

      // // 将所有文件名存入 pathList
      this.pathList = files.filter(file =>
      (file.endsWith('.jpg') || file.endsWith('.png'))
      );

    } catch (err) {
      hilog.error(0x0000, TAG, `Error reading files: ${err.message}`);
    }
  }
  private async segmentAllImagesInSandbox() {
    for (let path of this.pathList) {
      let fullPath = getContext().filesDir + '/' + path;
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
        let fileName = path.split('/').pop() || 'unknown.jpg'
        let segmentedName = 'segmented_' + fileName;
        let savePath = getContext().filesDir + '/' + segmentedName;
        hilog.info(0x0000, TAG, `目标路径: ${savePath}`);
        let file = fs.openSync(savePath, fs.OpenMode.CREATE | fs.OpenMode.READ_WRITE);
        await image.createImagePacker().packToFile(data.fullSubject.foregroundImage, file.fd, {
          format: "image/jpeg",
          quality: 100
        });
        fs.closeSync(file);
        hilog.info(0x0000, TAG, `保存分割图: ${savePath}`);
      } else {
        hilog.warn(0x0000, TAG, `图像 ${path} 分割无前景图`);
      }
    }
  }



}
