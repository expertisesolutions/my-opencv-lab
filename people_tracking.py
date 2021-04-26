import cv2  # or opencv-python
import numpy as np
import time


class PersonTracker:
    def __init__(
        self,
        id,
        frame,
        bbox,
        tracking_algorithm,
        fails_limit,
        color=(0, 255, 0),
        debug=False,
    ):
        self.debug = debug
        self.fails_limit = fails_limit

        bbox = tuple(bbox.astype(int))

        # Select our tracking algorithm and create our multi tracker
        OPENCV_OBJECT_TRACKERS = {
            # "boosting": cv2.TrackerBoosting_create,  # opencv 3.4
            "mil": cv2.TrackerMIL_create,
            "kcf": cv2.TrackerKCF_create,
            # "tld": cv2.TrackerTLD_create,  # opencv 3.4
            # "medianflow": cv2.TrackerMedianFlow_create,  # opencv 3.4
            "goturn": cv2.TrackerGOTURN_create,
            # "mosse": cv2.TrackerMOSSE_create,  # opencv 3.4
            "csrt": cv2.TrackerCSRT_create,
        }
        self.tracker = OPENCV_OBJECT_TRACKERS[tracking_algorithm]()
        self.tracker.init(frame, bbox)
        self.active, bbox = self.tracker.update(frame)
        bbox = tuple(np.array(bbox, dtype=int))

        if self.active:
            self.tracking_algorithm = tracking_algorithm
            self.id = int(id)
            self.centroid = self.get_centroid(bbox)
            self.fails = int(0)
            self.color = color
            self.bbox = bbox

    def __str__(self):
        return f"id: {self.id}, fails: {self.fails}, active: {self.active}, bbox: {self.bbox}"

    def update(self, frame):
        if not self.active:
            return False

        retval, bbox = self.tracker.update(frame)
        bbox = tuple(np.array(bbox, dtype=int))

        stucked = (self.bbox[0] == bbox[0]) and (self.bbox[1] == bbox[1])
        if (retval is False) or (stucked is True):
            self.fails += int(1)
        else:
            self.fails = int(0)
            self.active = True
        self.bbox = bbox

        print(
            f"Updated id {self.id}, fails: {self.fails}, bbox: {self.bbox} -> {bbox}"
        )
        if self.fails >= self.fails_limit:
            self.remove()

        return self.active

    def remove(self):
        self.active = False

    def get_centroid(self, bbox):
        (x, y, w, h) = bbox
        xc = int(x + (w * 0.5))
        yc = int(y + (h * 0.5))
        return xc, yc

    def draw(self, frame):
        (x, y, w, h) = np.array(self.bbox, dtype=int)
        cv2.rectangle(
            frame,
            pt1=(x, y),
            pt2=(x + w, y + h),
            color=self.color,
            thickness=1,
        )
        centroid = self.get_centroid(self.bbox)
        cv2.circle(
            frame, center=centroid, radius=2, color=self.color, thickness=1
        )
        cv2.putText(
            frame,
            text=f"{(int(self.id))}/{int(self.fails)}",
            org=(x, y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.4,
            color=self.color,
            thickness=1,
        )


class PeopleTracker:
    def __init__(self, debug=False):
        self.debug = debug
        self.trackers = list()

    def __str__(self):
        ret = ""
        for i in range(len(self.trackers)):
            ret += f"{self.trackers[i]}\n"
        return ret[:-1]

    def update(self, frame):
        if len(self.trackers) == 0:
            return False
        for i in range(len(self.trackers)):
            retval = self.trackers[i].update(frame)

    def isPointInsideRect(self, point, rect) -> bool:
        (x, y) = point
        (x1, y1, w, h) = rect
        (x2, y2) = (x1 + w, y1 + h)
        return (x1 < x < x2) and (y1 < y < y2)

    def add(self, frame, bbox, tracking_algorithm="kcf", fails_limit=25):
        id = len(self.trackers) + 1

        tracker = PersonTracker(
            id, frame, bbox, tracking_algorithm, fails_limit
        )

        # Here is the integration between Detection and Tracking:
        # We are only adding a new person if its centroid resides outside
        # any other active tracked person's bbox, otherwise we use that
        # detecion to update the already tracked person. Note that this
        # is not the best idea to deal with oclusion.
        isNew = True
        if tracker.active:
            for i in range(len(self.trackers)):
                if self.isPointInsideRect(
                    tracker.centroid, self.trackers[i].bbox
                ):
                    # Reset their fails
                    self.trackers[i].fails = 0
                    # Reactivate with this new tracker if it is inactive
                    if not self.trackers[i].active:
                        id = i
                        tracker.id = id
                        self.trackers[i].tracker = tracker
                    isNew = False

        if isNew:
            print(f"New person tracker added with id {id}.")
            self.trackers.append(tracker)

        return id

    def remove(self, id):
        print(f"Person tracker with id {id} removed.")
        self.trackers[id].remove()

    def draw(self, frame):
        if len(self.trackers) == 0:
            return False
        for i in range(len(self.trackers)):
            if self.trackers[i].active:
                self.trackers[i].draw(frame)


def HaarCascadeTracker(
    vcap, frames_to_process, minSize, maxSize, keyframe_interval
):
    # Create our body classifier
    detector = cv2.CascadeClassifier(
        # cv2.data.haarcascades + 'haarcascade_fullbody.xml'
        cv2.data.haarcascades
        + "haarcascade_upperbody.xml"
    )

    frame_count = 0
    processed_frames = np.zeros(frames_to_process, dtype=object)
    fps_timer = [0, cv2.getTickCount()]

    green = (0, 255, 0)
    red = (255, 0, 0)

    # Create our People Tracker
    people = PeopleTracker()

    while vcap.isOpened():
        # Read a frame
        retval, frame = vcap.read()
        if not retval or frame_count == frames_to_process:
            break

        # Use the classifier to detect new people
        if frame_count % keyframe_interval == 0:
            bboxes = detector.detectMultiScale(
                frame,
                scaleFactor=1.05,
                minNeighbors=2,
                minSize=minSize,
                maxSize=maxSize,
            )

            for bbox in bboxes:
                people.add(
                    frame,
                    bbox,
                    fails_limit=50,
                    tracking_algorithm="kcf",
                    # tracking_algorithm="csrt",
                    # tracking_algorithm="mil",
                    # tracking_algorithm="goturn",
                )

        people.update(frame)

        people.draw(frame)

        # Compute and put FPS on frame
        fps = cv2.getTickFrequency() / (fps_timer[1] - fps_timer[0])
        fps_timer[0] = fps_timer[1]
        fps_timer[1] = cv2.getTickCount()
        cv2.putText(
            frame,
            text=f"FPS: {int(fps)}",
            org=(frame_width - 60, frame_height - 5),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.3,
            color=green,
            thickness=1,
        )

        # Save frame
        processed_frames[frame_count] = frame
        frame_count += 1

        # Show in app
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)

    return (frame_count, processed_frames)


if __name__ == "__main__":
    print("Available Trackers:")
    for d in dir(cv2):
        if "Tracker" in d:
            print("\t -", d)

    # Open the input video capture
    # minSize, maxSize, input_filename = ((100,100), (120,120), './1080p_TownCentreXVID.mp4')
    # minSize, maxSize, input_filename = (None, (120,120), './720p_TownCentreXVID.mp4')
    # minSize, maxSize, input_filename = ((20,20), (80,80), './480p_TownCentreXVID.mp4')
    minSize, maxSize, input_filename = (
        (5, 10),
        (60, 40),
        "./360p_TownCentreXVID.mp4",
    )
    vcap = cv2.VideoCapture(input_filename)

    # Get video properties
    frame_width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vcap.get(cv2.CAP_PROP_FPS)
    n_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("Frame width:", frame_width)
    print("Frame width:", frame_height)
    print("Video fps:", fps)

    # Setup the output video file
    output_filename = "./output.mp4"
    apiPreference = cv2.CAP_FFMPEG
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vout = cv2.VideoWriter(
        filename=output_filename,
        apiPreference=apiPreference,
        fourcc=fourcc,
        fps=fps,
        frameSize=(frame_width, frame_height),
    )

    print(f'Processing "{input_filename}" ({int(n_frames)} frames)...')

    # # Start app
    window_name = "People Tracking"
    cv2.startWindowThread()
    cv2.namedWindow(window_name)

    # Loop each frame
    frames_to_process = 1000

    # start timer
    start = time.time()
    keyframe_interval = 10

    # Tracker Function
    frame_count, processed_frames = HaarCascadeTracker(
        vcap, frames_to_process, minSize, maxSize, keyframe_interval
    )

    # end timer
    end = time.time()
    overall_elapsed_time = end - start
    elapsed_time_per_frame = overall_elapsed_time / frame_count

    print("Done!")
    print(f"{frame_count} frames processed in {overall_elapsed_time} seconds.")
    print(f"({elapsed_time_per_frame}) seconds per frame.")
    print(f"({1/elapsed_time_per_frame}) frames per second.")

    # Write processed frames to file
    for frame in processed_frames:
        vout.write(frame)

    print(f'Output saved to "{output_filename}".')

    vcap.release()
    vout.release()
    cv2.destroyAllWindows()
