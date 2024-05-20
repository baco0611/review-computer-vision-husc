import cv2

def keypoints_to_data(list_of_keypoint_tuples):
    """Chuyển đổi list gồm các tuple keypoints sang dạng có thể serialize."""
    result = []
    for keypoints in list_of_keypoint_tuples:
        # Đối với mỗi tuple chứa keypoints, chuyển đổi và thêm vào kết quả
        converted = [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in keypoints]
        result.append(converted)
    return result

def data_to_keypoints(list_of_converted_tuples):
    """Chuyển đổi từ dữ liệu đã serialize trở lại thành list các tuple keypoints."""
    result = []
    for converted in list_of_converted_tuples:
        # Đối với mỗi tuple đã được chuyển đổi, chuyển đổi trở lại thành keypoints
        keypoints = [cv2.KeyPoint(x=pt[0], y=pt[1], _size=size, _angle=angle, _response=response, _octave=octave, _class_id=class_id) 
                     for pt, size, angle, response, octave, class_id in converted]
        result.append(keypoints)
    return result