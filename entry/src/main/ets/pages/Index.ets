import { hilog } from '@kit.PerformanceAnalysisKit';
import nativeRender from 'libnativerender.so';
import CommonConstants from '../common/CommonConstants';
import image from '@ohos.multimedia.image'
import { BusinessError } from '@kit.BasicServicesKit';
const context = getContext(this);
const filesDir = context.filesDir;

@Entry
@Component
struct Index {
  private xComponentContext: Record<string, (a: number | boolean | undefined) => void> = {};
  private resmgr = getContext().resourceManager;

  @State progressValue: number = 0;
  @State message: string = "3DGS Training";
  @State isTraining: boolean = false;
  @State pixelMap : PixelMap | undefined = undefined;
  @State currentImage: number = 0;

  aboutToAppear() {
    // 调用自定义的getSyncPixelMap接口，获取pixelMap
    // this.pixelMap = nativeRender.getSyncPixelMap(this.resmgr, "assets/case1_head/images/15506175713000.jpg")
    this.pixelMap = nativeRender.getSyncPixelMap(this.resmgr, "assets/case2/test/0.jpg")

  }

  async train3DGS(inputPath: string) {
    if (this.isTraining) return;
      this.isTraining = true;
    try {
      nativeRender.initTrainProgress((progress: number) => {
        this.progressValue = progress;
      });
      await nativeRender.train3DGS(this.resmgr, inputPath);
      this.message = "Training Complete!";
    } catch (error) {
      // Handle error
    }
  }

  async switchImage(){
    this.pixelMap = nativeRender.getSandBoxPixelMap(filesDir, "/temp/" + this.currentImage.toString() + ".jpg")
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
              this.train3DGS("/assets/case2");
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
}