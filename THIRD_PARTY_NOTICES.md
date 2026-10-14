# Third-Party Software and Model Notices

This document details third-party software, libraries, and machine learning models integrated or referenced by **LUMI Vision**.

---

## 1. Computer Vision & Recognition Models

### A. YuNet Face Detector
* **Project:** YuNet (via OpenCV Zoo)
* **Author / Maintainers:** Shiqi Yu and OpenCV contributors
* **Repository:** [https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
* **Model Asset:** `face_detection_yunet_2023mar.onnx`
* **Upstream Commit / Tag:** 2023 March release
* **License:** Apache License, Version 2.0
* **Model Card:** [https://huggingface.co/opencv/face_detection_yunet](https://huggingface.co/opencv/face_detection_yunet)
* **Local Usage:** Embedded as the primary real-time face detector via OpenCV's `cv2.FaceDetectorYN` C++/Python binding.

### B. SFace Face Recognizer
* **Project:** SFace (via OpenCV Zoo)
* **Author / Reference:** Zhong et al., *SphereFace2: Binary Classification is All You Need for Deep Face Recognition*
* **Repository:** [https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)
* **Model Asset:** `face_recognition_sface_2021dec.onnx`
* **Upstream Commit / Tag:** 2021 December release
* **License:** Apache License, Version 2.0
* **Model Card:** [https://huggingface.co/opencv/face_recognition_sface](https://huggingface.co/opencv/face_recognition_sface)
* **Local Usage:** Aligns faces using YuNet landmarks and computes 128-dimensional L2-normalized embeddings via `cv2.FaceRecognizerSF`.

---

## 2. Python Libraries

### A. OpenCV (opencv-python)
* **Project:** Open Source Computer Vision Library
* **License:** Apache License, Version 2.0
* **Homepage:** [https://opencv.org/](https://opencv.org/)
* **Repository:** [https://github.com/opencv/opencv](https://github.com/opencv/opencv)

### B. NumPy
* **Project:** NumPy Array and Numerical Computation Library
* **License:** BSD 3-Clause "New" or "Revised" License
* **Homepage:** [https://numpy.org/](https://numpy.org/)

### C. PyYAML
* **Project:** PyYAML YAML Parser and Emitter for Python
* **License:** MIT License
* **Homepage:** [https://pyyaml.org/](https://pyyaml.org/)

### D. pytest
* **Project:** pytest Testing Framework
* **License:** MIT License
* **Homepage:** [https://pytest.org/](https://pytest.org/)

---

## 3. License Texts

### Apache License 2.0
```text
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

### BSD 3-Clause License
```text
Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.
```

### MIT License
```text
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```
