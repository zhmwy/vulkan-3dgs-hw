if (!("finalizeConstruction" in ViewPU.prototype)) {
    Reflect.set(ViewPU.prototype, "finalizeConstruction", () => { });
}
interface Index_Params {
    chooseImage?: PixelMap | undefined;
    dataValues?: string;
    segmentedImage?: PixelMap | undefined;
    maxNum?: string;
    selectedImagePath?: string | undefined;
    pathList?: string[];
    xComponentContext?: Record<string, (a: number | boolean | undefined) => void>;
    resmgr?;
    progressValue?: number;
    message?: string;
    isTraining?: boolean;
    pixelMap?: PixelMap | undefined;
    currentImage?: number;
    segmentViews?: number;
}
import hilog from "@ohos:hilog";
import nativeRender from "@app:com.huawei.xenginevulkandemo/entry/input";
import subjectSegmentation from "@hms:ai.vision.subjectSegmentation";
import image from "@ohos:multimedia.image";
import fileIo from "@ohos:file.fs";
import type { ListFileOptions } from "@ohos:file.fs";
import fs from "@ohos:file.fs";
const context = getContext(this);
const filesDir = context.filesDir;
const TAG: string = "ImageSegmentationSample";
let caseName = "2025-06-04_15_34_45";
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
];
const chosenCase: string[] = [
    "2025-06-04_15_34_45",
    "2025-06-04_19_40_24",
    "2025-06-04_20_26_45",
    "2025-06-05_20_27_20"
];
class GS_Mode {
    static readonly SCENE = 0; // 暂无特定优化
    static readonly OBJECT = 1; // 局部放大、EFA-GS
    static readonly HUMAN = 2; // SMPL估计、人脸重点、美颜
    static readonly MULTI_HUMAN = 3; // 人脸重点
}
const trainingMode: number = GS_Mode.OBJECT;
const isObject: boolean = false;
class Index extends ViewPU {
    constructor(parent, params, __localStorage, elmtId = -1, paramsLambda = undefined, extraInfo) {
        super(parent, __localStorage, elmtId, extraInfo);
        if (typeof paramsLambda === "function") {
            this.paramsGenerator_ = paramsLambda;
        }
        this.__chooseImage = new ObservedPropertyObjectPU(undefined, this, "chooseImage");
        this.__dataValues = new ObservedPropertySimplePU('', this, "dataValues");
        this.__segmentedImage = new ObservedPropertyObjectPU(undefined, this, "segmentedImage");
        this.__maxNum = new ObservedPropertySimplePU('20', this, "maxNum");
        this.__selectedImagePath = new ObservedPropertyObjectPU(undefined, this, "selectedImagePath");
        this.__pathList = new ObservedPropertyObjectPU([], this, "pathList");
        this.xComponentContext = {};
        this.resmgr = getContext().resourceManager;
        this.__progressValue = new ObservedPropertySimplePU(0, this, "progressValue");
        this.__message = new ObservedPropertySimplePU("3DGS Training", this, "message");
        this.__isTraining = new ObservedPropertySimplePU(false, this, "isTraining");
        this.__pixelMap = new ObservedPropertyObjectPU(undefined, this, "pixelMap");
        this.__currentImage = new ObservedPropertySimplePU(0, this, "currentImage");
        this.__segmentViews = new ObservedPropertySimplePU(8, this, "segmentViews");
        this.setInitiallyProvidedValue(params);
        this.finalizeConstruction();
    }
    setInitiallyProvidedValue(params: Index_Params) {
        if (params.chooseImage !== undefined) {
            this.chooseImage = params.chooseImage;
        }
        if (params.dataValues !== undefined) {
            this.dataValues = params.dataValues;
        }
        if (params.segmentedImage !== undefined) {
            this.segmentedImage = params.segmentedImage;
        }
        if (params.maxNum !== undefined) {
            this.maxNum = params.maxNum;
        }
        if (params.selectedImagePath !== undefined) {
            this.selectedImagePath = params.selectedImagePath;
        }
        if (params.pathList !== undefined) {
            this.pathList = params.pathList;
        }
        if (params.xComponentContext !== undefined) {
            this.xComponentContext = params.xComponentContext;
        }
        if (params.resmgr !== undefined) {
            this.resmgr = params.resmgr;
        }
        if (params.progressValue !== undefined) {
            this.progressValue = params.progressValue;
        }
        if (params.message !== undefined) {
            this.message = params.message;
        }
        if (params.isTraining !== undefined) {
            this.isTraining = params.isTraining;
        }
        if (params.pixelMap !== undefined) {
            this.pixelMap = params.pixelMap;
        }
        if (params.currentImage !== undefined) {
            this.currentImage = params.currentImage;
        }
        if (params.segmentViews !== undefined) {
            this.segmentViews = params.segmentViews;
        }
    }
    updateStateVars(params: Index_Params) {
    }
    purgeVariableDependenciesOnElmtId(rmElmtId) {
        this.__chooseImage.purgeDependencyOnElmtId(rmElmtId);
        this.__dataValues.purgeDependencyOnElmtId(rmElmtId);
        this.__segmentedImage.purgeDependencyOnElmtId(rmElmtId);
        this.__maxNum.purgeDependencyOnElmtId(rmElmtId);
        this.__selectedImagePath.purgeDependencyOnElmtId(rmElmtId);
        this.__pathList.purgeDependencyOnElmtId(rmElmtId);
        this.__progressValue.purgeDependencyOnElmtId(rmElmtId);
        this.__message.purgeDependencyOnElmtId(rmElmtId);
        this.__isTraining.purgeDependencyOnElmtId(rmElmtId);
        this.__pixelMap.purgeDependencyOnElmtId(rmElmtId);
        this.__currentImage.purgeDependencyOnElmtId(rmElmtId);
        this.__segmentViews.purgeDependencyOnElmtId(rmElmtId);
    }
    aboutToBeDeleted() {
        this.__chooseImage.aboutToBeDeleted();
        this.__dataValues.aboutToBeDeleted();
        this.__segmentedImage.aboutToBeDeleted();
        this.__maxNum.aboutToBeDeleted();
        this.__selectedImagePath.aboutToBeDeleted();
        this.__pathList.aboutToBeDeleted();
        this.__progressValue.aboutToBeDeleted();
        this.__message.aboutToBeDeleted();
        this.__isTraining.aboutToBeDeleted();
        this.__pixelMap.aboutToBeDeleted();
        this.__currentImage.aboutToBeDeleted();
        this.__segmentViews.aboutToBeDeleted();
        SubscriberManager.Get().delete(this.id__());
        this.aboutToBeDeletedInternal();
    }
    private __chooseImage: ObservedPropertyObjectPU<PixelMap | undefined>;
    get chooseImage() {
        return this.__chooseImage.get();
    }
    set chooseImage(newValue: PixelMap | undefined) {
        this.__chooseImage.set(newValue);
    }
    private __dataValues: ObservedPropertySimplePU<string>;
    get dataValues() {
        return this.__dataValues.get();
    }
    set dataValues(newValue: string) {
        this.__dataValues.set(newValue);
    }
    private __segmentedImage: ObservedPropertyObjectPU<PixelMap | undefined>;
    get segmentedImage() {
        return this.__segmentedImage.get();
    }
    set segmentedImage(newValue: PixelMap | undefined) {
        this.__segmentedImage.set(newValue);
    }
    private __maxNum: ObservedPropertySimplePU<string>;
    get maxNum() {
        return this.__maxNum.get();
    }
    set maxNum(newValue: string) {
        this.__maxNum.set(newValue);
    }
    private __selectedImagePath: ObservedPropertyObjectPU<string | undefined>; // 新增
    get selectedImagePath() {
        return this.__selectedImagePath.get();
    }
    set selectedImagePath(newValue: string | undefined) {
        this.__selectedImagePath.set(newValue);
    }
    private __pathList: ObservedPropertyObjectPU<string[]>;
    get pathList() {
        return this.__pathList.get();
    }
    set pathList(newValue: string[]) {
        this.__pathList.set(newValue);
    }
    private xComponentContext: Record<string, (a: number | boolean | undefined) => void>;
    private resmgr;
    private __progressValue: ObservedPropertySimplePU<number>;
    get progressValue() {
        return this.__progressValue.get();
    }
    set progressValue(newValue: number) {
        this.__progressValue.set(newValue);
    }
    private __message: ObservedPropertySimplePU<string>;
    get message() {
        return this.__message.get();
    }
    set message(newValue: string) {
        this.__message.set(newValue);
    }
    private __isTraining: ObservedPropertySimplePU<boolean>;
    get isTraining() {
        return this.__isTraining.get();
    }
    set isTraining(newValue: boolean) {
        this.__isTraining.set(newValue);
    }
    private __pixelMap: ObservedPropertyObjectPU<PixelMap | undefined>;
    get pixelMap() {
        return this.__pixelMap.get();
    }
    set pixelMap(newValue: PixelMap | undefined) {
        this.__pixelMap.set(newValue);
    }
    private __currentImage: ObservedPropertySimplePU<number>;
    get currentImage() {
        return this.__currentImage.get();
    }
    set currentImage(newValue: number) {
        this.__currentImage.set(newValue);
    }
    private __segmentViews: ObservedPropertySimplePU<number>;
    get segmentViews() {
        return this.__segmentViews.get();
    }
    set segmentViews(newValue: number) {
        this.__segmentViews.set(newValue);
    }
    aboutToAppear() {
        // 调用自定义的getSyncPixelMap接口，获取pixelMap
        // this.pixelMap = nativeRender.getSyncPixelMap(this.resmgr, "assets/12_data/AVATAR/images_orig/frame_0000.jpg")
        // this.pixelMap = nativeRender.getSyncPixelMap(this.resmgr, "assets/case1/images/15506175713000.jpg")
        this.pixelMap = nativeRender.getSyncPixelMap(this.resmgr, `${caseName}/images/85864565279156.jpg`);
        this.train3DGS(caseName);
    }
    async train3DGS(inputPath: string) {
        if (this.isTraining)
            return;
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
            caseName = caseNames[0];
            await this.readFiles(filesDir + "/" + caseName + "/images", this.segmentViews); // 调用 readFiles 读取文件kl;lkm
            hilog.info(0x0000, TAG, `文件列表已获取，沙盒内文件: ${this.pathList}`);
            if (trainingMode == GS_Mode.OBJECT) {
                await this.segmentAllImagesInSandbox();
            }
            await nativeRender.train3DGS(this.resmgr, caseName, GS_Mode.OBJECT, isObject); // for循环
            this.message = "Training Complete!";
        }
        catch (error) {
            // Handle error
        }
    }
    async switchImage() {
        this.pixelMap = nativeRender.getSandBoxPixelMap(filesDir, "/" + caseName + "/temp" + this.currentImage.toString() + ".jpg");
        this.currentImage = (this.currentImage + 1) % 1000;
    }
    initialRender() {
        this.observeComponentCreation2((elmtId, isInitialRender) => {
            Row.create();
            Row.debugLine("entry/src/main/ets/pages/Index.ets(126:5)", "entry");
        }, Row);
        this.observeComponentCreation2((elmtId, isInitialRender) => {
            Column.create();
            Column.debugLine("entry/src/main/ets/pages/Index.ets(127:7)", "entry");
            Column.width('100%');
            Column.height('100%');
        }, Column);
        this.observeComponentCreation2((elmtId, isInitialRender) => {
            Text.create(this.message);
            Text.debugLine("entry/src/main/ets/pages/Index.ets(128:9)", "entry");
            Text.fontSize(50);
            Text.fontWeight(FontWeight.Bold);
            Text.margin(10);
        }, Text);
        Text.pop();
        this.observeComponentCreation2((elmtId, isInitialRender) => {
            // Button(this.isTraining ? 'Training...' : 'Start Training')
            //   .onClick(() => {
            //     if (!this.isTraining) {
            //       this.train3DGS(caseName);
            //     }
            //   })
            //   .margin(10)
            //   .enabled(!this.isTraining)
            Progress.create({ value: this.progressValue, total: 100, type: ProgressType.Ring });
            Progress.debugLine("entry/src/main/ets/pages/Index.ets(142:9)", "entry");
            // Button(this.isTraining ? 'Training...' : 'Start Training')
            //   .onClick(() => {
            //     if (!this.isTraining) {
            //       this.train3DGS(caseName);
            //     }
            //   })
            //   .margin(10)
            //   .enabled(!this.isTraining)
            Progress.width(100);
            // Button(this.isTraining ? 'Training...' : 'Start Training')
            //   .onClick(() => {
            //     if (!this.isTraining) {
            //       this.train3DGS(caseName);
            //     }
            //   })
            //   .margin(10)
            //   .enabled(!this.isTraining)
            Progress.height(100);
            // Button(this.isTraining ? 'Training...' : 'Start Training')
            //   .onClick(() => {
            //     if (!this.isTraining) {
            //       this.train3DGS(caseName);
            //     }
            //   })
            //   .margin(10)
            //   .enabled(!this.isTraining)
            Progress.margin(20);
        }, Progress);
        Column.pop();
        Row.pop();
    }
    private async readFiles(path: string, numToSelect?: number) {
        try {
            // 调用 listFile 获取目录下所有文件
            let options: ListFileOptions = {
                recursion: false, // 不递归子目录
            };
            hilog.info(0x0000, TAG, `正在读取 ${path}中的文件`);
            // 获取文件列表
            let files = await fs.listFile(path, options);
            // 过滤出所有图片文件
            let allImages = files.filter(file => (file.endsWith('.jpg') || file.endsWith('.png')));
            // 如果指定了选择数量，则均匀选择N张图片
            if (numToSelect && numToSelect > 0 && numToSelect < allImages.length) {
                this.pathList = this.selectImagesEvenly(allImages, numToSelect);
                hilog.info(0x0000, TAG, `从 ${allImages.length} 张图片中均匀选择了 ${this.pathList.length} 张`);
            }
            else {
                // 否则使用所有图片
                this.pathList = allImages;
                hilog.info(0x0000, TAG, `使用所有 ${this.pathList.length} 张图片`);
            }
        }
        catch (err) {
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
        hilog.info(0x0000, TAG, `开始分割 ${this.pathList.length} 张图片`);
        for (let i = 0; i < this.pathList.length; i++) {
            let path = this.pathList[i];
            let fullPath = getContext().filesDir + `/${caseName}/images/` + path;
            hilog.info(0x0000, TAG, `正在分割第 ${i + 1}/${this.pathList.length} 张: ${path}`);
            let fileSource = await fileIo.open(fullPath, fileIo.OpenMode.READ_ONLY);
            let imageSource = image.createImageSource(fileSource.fd);
            let pixelMap = await imageSource.createPixelMap();
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
                hilog.info(0x0000, TAG, `已保存分割图 ${i + 1}/${this.pathList.length}: ${savePath}`);
            }
            else {
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
                binaryPixels[i] = 255; // R
                binaryPixels[i + 1] = 255; // G
                binaryPixels[i + 2] = 255; // B
                binaryPixels[i + 3] = 255; // A
            }
            else { // Background
                // Set to black
                binaryPixels[i] = 0; // R
                binaryPixels[i + 1] = 0; // G
                binaryPixels[i + 2] = 0; // B
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
    rerender() {
        this.updateDirtyElements();
    }
    static getEntryName(): string {
        return "Index";
    }
}
registerNamedRoute(() => new Index(undefined, {}), "", { bundleName: "com.huawei.xenginevulkandemo", moduleName: "entry", pagePath: "pages/Index", pageFullPath: "entry/src/main/ets/pages/Index", integratedHsp: "false" });
