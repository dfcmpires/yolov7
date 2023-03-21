#from detect import detect
import argparse
import time
from pathlib import Path
import pandas as pd
import ffmpeg

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import params

from fastapi import FastAPI, UploadFile, Response
from fastapi.responses import FileResponse
import argparse

import asyncio
import aiofiles

from os import path

app = FastAPI()

def count(founded_classes,im0):
  model_values=[]
  aligns=im0.shape
  align_bottom=aligns[0]
  align_right=(aligns[1]/1.7 )

  for i, (k, v) in enumerate(founded_classes.items()):
    a=f"{k} = {v}"
    model_values.append(v)
    align_bottom=align_bottom-35
    cv2.putText(im0, str(a) ,(int(align_right),align_bottom), cv2.FONT_HERSHEY_SIMPLEX, 1,(45,255,255),1,cv2.LINE_AA)


def detect(opt, save_img=False, ):

    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave #and not source.endswith('.txt')  # save inference images
    webcam = False #source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
    #    ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    number_of_frames = getattr(dataset, 'nframes', 0)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    # Starting time (for measuring the total inference time)
    t0 = time.time()


    # Initialize total count
    max_categories_counts = {name: [] for name in names}
    cummulative_categories_counts = {name: [] for name in names}

    # Go through each image/frame (in video... )
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference: t2 - t1 (in ms)
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0] # TODO: understand what information is in each prediction
        t2 = time_synchronized()

        # Apply NMS (TODO: understand what this does): t3 - t2 (in ms)
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier (TODO: understand the need)
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)


        # Initialize counts of each category for this frame (to populate given the predictions)
        categories_count = {name: 0 for name in names}

        # Process detections (each image might have multiple detections)
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                # Set attributes for each bbox in the image
                    # p is path to the image;
                    # s is the string to print (currently empty)
                    # im0s is ???
                    # frame is the size of image (same for every image)
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh


            # do what you need with the predictions
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()


                founded_classes={}
                # Print results and story category counts
                for category in det[:, -1].unique(): # last axis of det is the category
                    category_name = names[int(category)]
                    n = (det[:, -1] == category).sum()  # detections per category

                    class_index=int(category) #counting display
                    founded_classes[names[class_index]]=int(n)  #counting display

                    s += f"{n} {category_name}{'s' * (n > 1)}, "  # add to string (e.g. 3 fish)
                    categories_count[category_name] = n # add to dictionary (e.g. {'fish': 3})

                    count(founded_classes=founded_classes,im0=im0) #counting display


                # Write each bbox in image
                for *xyxy, conf, category in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (category, *xywh, conf) if opt.save_conf else (category, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # Add bbox to image
                    if save_img or view_img:
                        label = f'{names[int(category)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(category)], line_thickness=1)
            else:
                s += "No detections, "
            # Print time (inference + NMS)
            print(f'Frame {frame} of {number_of_frames}: {s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Add counts to the total count
            for key, value in categories_count.items():
                cummulative_categories_counts[key].append(value)

            if frame % params.COUNT_EVERY_N_FRAMES == 0:
                # calculate max for each class
                max_counts = {key: max(values) for key, values in cummulative_categories_counts.items()}
                # print the max counts
                print(f"Max counts for each class: {max_counts}")

                # save max counts
                for key, value in max_counts.items():
                    max_categories_counts[key].append(value)

                #sum of max counts
                total_max_counts = {key: sum(values) for key, values in max_categories_counts.items()}

                # clear the cummulative counts for next round
                cummulative_categories_counts = {key: [] for key in cummulative_categories_counts.keys()}



            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    print(f'Done. ({time.time() - t0:.3f}s)')

    #print the total objects per each class
    for category_name, total_max_count in total_max_counts.items():
        print(f"Total counting: {category_name} = {total_max_count}")

    #save number of total objects per each class in a csv file

    df = pd.DataFrame(list(total_max_counts.items()), columns=["Classes", "Count"])

# Extract numerical value from tensor object in "Count" column
    df["Count"] = df["Count"].apply(lambda x: x.item() if hasattr(x, "item") else x)

# Save dataframe to a csv file without the index
    df.to_csv("output.csv", index=False)

    ##

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    return save_path

def prediction(source, confthres=0.25):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model  path(s)')
    parser.add_argument('--source', type=str, default=source, help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=confthres, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args(args=[])
    results = detect(opt)
    return results


@app.post("/predict")
async def predict_endpoint(file: UploadFile):
    confthres = file.headers.get("confthres", 0.25)

    async with aiofiles.open(file.filename, 'wb') as out_file:
        while content := await file.read(1024):  # async read chunk
            await out_file.write(content)  # async write chunk

    video_path = prediction(file.filename, confthres)
    fishes = {}

    with open("output.csv", "r") as f:
        for line in f:
            key, value = line.split(",")
            key = key.strip('\n')
            fishes[key] = str(value).strip('\n')
    header = {'X-Fishes' : str(fishes)}

    print(video_path)


    input_file = video_path
    output_file = video_path.replace(".mp4", "_compressed.mp4")

    input_stream = ffmpeg.input(input_file)
    output_stream = ffmpeg.output(input_stream, output_file, vcodec='libx264', crf=23, preset='veryfast')

    ffmpeg.run(output_stream)


    return FileResponse(output_file, media_type="video/mp4", headers=header)

if __name__ == "__main__":
    prediction("Savel.mp4")
