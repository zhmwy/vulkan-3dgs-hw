#include "video2images.h"


// Function to save a frame as an image
bool saveFrameAsImage(AVFrame* frame, const std::string& filename, int width, int height) {
    // Check file extension
    std::string ext = filename.substr(filename.find_last_of("."));
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    // Use MJPEG encoding for image output
    const AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_MJPEG);
    if (!codec) {
        LOGE("MJPEG codec not found");
        return false;
    }
    
    AVCodecContext* codecContext = avcodec_alloc_context3(codec);
    if (!codecContext) {
        LOGE("Could not allocate codec context");
        return false;
    }
    
    codecContext->width = width;
    codecContext->height = height;
    codecContext->time_base = {1, 25};
    codecContext->pix_fmt = AV_PIX_FMT_YUVJ420P;  // JPEG format
    codecContext->qmin = 1;  // High quality
    codecContext->qmax = 1;
    
    if (avcodec_open2(codecContext, codec, nullptr) < 0) {
        avcodec_free_context(&codecContext);
        return false;
    }
    
    // Convert RGB to YUV for JPEG
    AVFrame* yuvFrame = av_frame_alloc();
    if (!yuvFrame) {
        avcodec_free_context(&codecContext);
        return false;
    }
    
    yuvFrame->width = width;
    yuvFrame->height = height;
    yuvFrame->format = AV_PIX_FMT_YUVJ420P;
    
    if (av_frame_get_buffer(yuvFrame, 0) < 0) {
        av_frame_free(&yuvFrame);
        avcodec_free_context(&codecContext);
        return false;
    }
    
    SwsContext* swsContext = sws_getContext(
        width, height, AV_PIX_FMT_RGB24,
        width, height, AV_PIX_FMT_YUVJ420P,
        SWS_BILINEAR, nullptr, nullptr, nullptr);
    
    if (!swsContext) {
        av_frame_free(&yuvFrame);
        avcodec_free_context(&codecContext);
        return false;
    }
    
    if (sws_scale(swsContext, frame->data, frame->linesize, 0, height,
                 yuvFrame->data, yuvFrame->linesize) <= 0) {
        sws_freeContext(swsContext);
        av_frame_free(&yuvFrame);
        avcodec_free_context(&codecContext);
        return false;
    }
    
    sws_freeContext(swsContext);
    
    // Encode frame to JPEG
    AVPacket* packet = av_packet_alloc();
    if (!packet) {
        av_frame_free(&yuvFrame);
        avcodec_free_context(&codecContext);
        return false;
    }
    
    if (avcodec_send_frame(codecContext, yuvFrame) < 0) {
        av_packet_free(&packet);
        av_frame_free(&yuvFrame);
        avcodec_free_context(&codecContext);
        return false;
    }
    
    int ret = avcodec_receive_packet(codecContext, packet);
    if (ret < 0) {
        av_packet_free(&packet);
        av_frame_free(&yuvFrame);
        avcodec_free_context(&codecContext);
        return false;
    }
    
    // Write JPEG data to file
    FILE* f = fopen(filename.c_str(), "wb");
    if (!f) {
        av_packet_free(&packet);
        av_frame_free(&yuvFrame);
        avcodec_free_context(&codecContext);
        return false;
    }
    
    fwrite(packet->data, 1, packet->size, f);
    fclose(f);
    
    // Clean up
    av_packet_free(&packet);
    av_frame_free(&yuvFrame);
    avcodec_free_context(&codecContext);
    
    return true;
}

// Get the rotation angle from video metadata
int getVideoRotation(AVStream* stream) {
    AVDictionaryEntry* rotateTag = av_dict_get(stream->metadata, "rotate", nullptr, 0);
    if (rotateTag) {
        // Rotation specified in the metadata
        int angle = std::stoi(rotateTag->value);
        return angle;
    }
    
    // Check for display matrix
    uint8_t* displayMatrix = av_stream_get_side_data(stream, AV_PKT_DATA_DISPLAYMATRIX, nullptr);
    if (displayMatrix) {
        double rotation = av_display_rotation_get((int32_t*)displayMatrix);
        if(rotation < 0){
            return 360 + (int)rotation;
        }
        return (int)(rotation);
    }
    
    return 0; // No rotation
}

// Process video with FFmpeg
bool processVideoWithFFmpeg(
    const std::string& videoPath, 
    const std::vector<std::string>& timestamps,
    const std::unordered_set<std::string>& uniqueImages,
    const std::string& scenepath,
    const std::string& ext,
    int width, int height,
    std::vector<std::string>& img_list) 
{
    // Initialize FFmpeg resources
    AVFormatContext* formatContext = nullptr;
    AVCodecContext* codecContext = nullptr;
    const AVCodec* codec = nullptr;
    AVFrame* frame = nullptr;
    AVFrame* rgbFrame = nullptr;
    AVPacket* packet = nullptr;
    SwsContext* swsContext = nullptr;
    uint8_t* buffer = nullptr;
    
    try {
        // Open input file
        if (avformat_open_input(&formatContext, videoPath.c_str(), nullptr, nullptr) != 0) {
            throw std::runtime_error("Could not open video file");
        }
        
        // Retrieve stream information
        if (avformat_find_stream_info(formatContext, nullptr) < 0) {
            throw std::runtime_error("Could not find stream information");
        }
        
        // Find the video stream
        int videoStreamIndex = -1;
        for (unsigned int i = 0; i < formatContext->nb_streams; i++) {
            if (formatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                videoStreamIndex = i;
                break;
            }
        }
        
        if (videoStreamIndex == -1) {
            throw std::runtime_error("Could not find video stream");
        }
        
        // Get video rotation
        int rotation = getVideoRotation(formatContext->streams[videoStreamIndex]);
        LOGI("Video rotation angle: %{public}d degrees", rotation);
        
        // Get codec parameters
        AVCodecParameters* codecParameters = formatContext->streams[videoStreamIndex]->codecpar;
        
        // Find decoder
        codec = avcodec_find_decoder(codecParameters->codec_id);
        if (!codec) {
            throw std::runtime_error("Unsupported codec");
        }
        
        // Create codec context
        codecContext = avcodec_alloc_context3(codec);
        if (!codecContext) {
            throw std::runtime_error("Failed to allocate codec context");
        }
        
        // Copy codec parameters to codec context
        if (avcodec_parameters_to_context(codecContext, codecParameters) < 0) {
            throw std::runtime_error("Failed to copy codec parameters to codec context");
        }
        
        // Open codec
        if (avcodec_open2(codecContext, codec, nullptr) < 0) {
            throw std::runtime_error("Failed to open codec");
        }
        
        // Allocate frames
        frame = av_frame_alloc();
        rgbFrame = av_frame_alloc();
        if (!frame || !rgbFrame) {
            throw std::runtime_error("Failed to allocate frames");
        }

        LOGI("Original height: %{public}d, Original width %{public}d", codecContext->height, codecContext->width);

        // Allocate RGB frame buffer with scaling dimensions
        int numBytes = av_image_get_buffer_size(AV_PIX_FMT_BGR24, codecContext->width, codecContext->height, 1);
        buffer = (uint8_t*)av_malloc(numBytes * sizeof(uint8_t));
        if (!buffer) {
            throw std::runtime_error("Failed to allocate buffer");
        }
        
        // Setup RGB frame with the buffer
        if (av_image_fill_arrays(rgbFrame->data, rgbFrame->linesize, buffer,
                                AV_PIX_FMT_RGB24, codecContext->width, codecContext->height, 1) < 0) {
            throw std::runtime_error("Failed to setup RGB frame");
        }

        // Initialize scaling context with rotation handling
        int swsFlags = SWS_BILINEAR;
        swsContext = nullptr;
        
        // Apply correct rotation during scaling
        swsContext = sws_getContext(
            codecContext->width, codecContext->height, codecContext->pix_fmt,
            codecContext->width, codecContext->height, AV_PIX_FMT_BGR24,
            swsFlags | SWS_FULL_CHR_H_INT, nullptr, nullptr, nullptr);
        
        if (!swsContext) {
            throw std::runtime_error("Failed to initialize scaling context");
        }
        
        // Allocate packet
        packet = av_packet_alloc();
        if (!packet) {
            throw std::runtime_error("Failed to allocate packet");
        }
        
        // Create output directory
        std::filesystem::create_directories(scenepath + "/images/");
        std::filesystem::create_directories(scenepath + "/images_orig/");
        
        // Read frames from the video
        int frameCount = 0;
        while (av_read_frame(formatContext, packet) >= 0) {
            if (packet->stream_index == videoStreamIndex) {
                // Send packet to decoder
                int response = avcodec_send_packet(codecContext, packet);
                if (response < 0) {
                    throw std::runtime_error("Error sending packet to decoder");
                }
                
                while (response >= 0) {
                    response = avcodec_receive_frame(codecContext, frame);
                    if (response == AVERROR(EAGAIN) || response == AVERROR_EOF) {
                        break;
                    } else if (response < 0) {
                        throw std::runtime_error("Error during decoding");
                    }
                    
                    // If we have a valid frame and it matches our timestamps
                    if (frameCount < timestamps.size()) {
                        std::string timestamp = timestamps[frameCount];
                        if (uniqueImages.find(timestamp + ext) != uniqueImages.end()) {

                            sws_scale(swsContext, frame->data, frame->linesize, 0, codecContext->height, rgbFrame->data, rgbFrame->linesize);
                            cv::Mat img(codecContext->height, codecContext->width, CV_8UC3, rgbFrame->data[0]);
                            cv::Mat rotated;
                            
                            if (rotation == 270) 
                                cv::rotate(img, rotated, cv::ROTATE_90_CLOCKWISE);
                            else if (rotation == 90) 
                                cv::rotate(img, rotated, cv::ROTATE_90_COUNTERCLOCKWISE);
                            else if (rotation == 180) 
                                cv::rotate(img, rotated, cv::ROTATE_180);
                            else
                                rotated = img;
                            std::string image_orig_Path = scenepath + "/images_orig/" + timestamp + ext;
                            bool success = cv::imwrite(image_orig_Path, rotated);
                            LOGI("Saving original image: %{public}s", image_orig_Path.c_str());
                            cv::resize(rotated, rotated, cv::Size(width, height));
                            std::string imagePath = scenepath + "/images/" + timestamp + ext;
                            img_list.push_back(imagePath);
                            cv::imwrite(imagePath, rotated);
                        }
                        frameCount++;
                    }
                }
            }
            av_packet_unref(packet);
        }
        std::cout << frameCount << std::endl;
    } catch (const std::exception& e) {
        LOGE("Error processing video: %s", e.what());
        // Cleanup FFmpeg resources
        if (buffer) av_free(buffer);
        if (swsContext) sws_freeContext(swsContext);
        if (frame) av_frame_free(&frame);
        if (rgbFrame) av_frame_free(&rgbFrame);
        if (packet) av_packet_free(&packet);
        if (codecContext) avcodec_free_context(&codecContext);
        if (formatContext) avformat_close_input(&formatContext);
        
        return false;
    }
    
    // Cleanup FFmpeg resources

    if (buffer) av_free(buffer);
    if (swsContext) sws_freeContext(swsContext);
    if (frame) av_frame_free(&frame);
    if (rgbFrame) av_frame_free(&rgbFrame);
    if (packet) av_packet_free(&packet);
    if (codecContext) avcodec_free_context(&codecContext);
    if (formatContext) avformat_close_input(&formatContext);
    
    return true;
}