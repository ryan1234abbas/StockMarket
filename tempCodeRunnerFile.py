    def run(self):
        
        def key_pressed():
            return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

        sys.stdin = open('/dev/tty')  # Ensure terminal input
        zoom = 0.6
        try:
            while self.running:
                if key_pressed():
                    key = sys.stdin.readline().strip().lower()
                    if key == 'q':
                        print("Stopping on 'q' pressed")
                        self.running = False
                        break
            
                start_time = time.time()

                # LEFT REGION
                lw_orig = self.width // 2
                lh_orig = self.height
                lw_zoom = int(lw_orig / zoom)
                lh_zoom = int(lh_orig / zoom)
                left_monitor = {
                    "top": self.offset_y - (lh_zoom - lh_orig) // 2,
                    "left": (self.offset_x - (lw_zoom - lw_orig) // 2) + 20,
                    "width": lw_zoom - 70,
                    "height": lh_zoom
                }

                # RIGHT REGION
                rw_orig = self.width - lw_orig
                rh_orig = self.height
                rw_zoom = int(rw_orig / zoom)
                rh_zoom = int(rh_orig / zoom)
                right_monitor = {
                    "top": self.offset_y - (rh_zoom - rh_orig) // 2,
                    "left": (self.offset_x + lw_orig - (rw_zoom - rw_orig) // 2) + 180,
                    "width": rw_zoom - 100,
                    "height": rh_zoom
                }

                left_img = np.array(self.sct.grab(left_monitor))[:, :, :3]
                right_img = np.array(self.sct.grab(right_monitor))[:, :, :3]

                m32 = lambda v: ((v + 31) // 32) * 32
                left_sz = (m32(left_monitor['width']), m32(left_monitor['height']))
                right_sz = (m32(right_monitor['width']), m32(right_monitor['height']))

                left_results = self.model.predict(
                    source=left_img, verbose=False, stream=False, conf=0.01, iou=0.15, imgsz=left_sz)
                right_results = self.model.predict(
                    source=right_img, verbose=False, stream=False, conf=0.01, iou=0.15, imgsz=right_sz)

                left_boxes, left_scores = self.process_results(left_results)
                right_boxes, right_scores = self.process_results(right_results)

                keep_left = self.non_max_suppression_fast(left_boxes, left_scores, iou_thresh=0.5)
                merged_left = self.merge_vertically_close_boxes([left_boxes[i] for i in keep_left])

                keep_right = self.non_max_suppression_fast(right_boxes, right_scores, iou_thresh=0.5)
                merged_right = self.merge_vertically_close_boxes([right_boxes[i] for i in keep_right])

                decision = self.analyze_candles_tm(left_img, merged_left, right_img, merged_right, self.templates)

                if decision:
                    print(f"Trade decision: {decision}")
                print(f"Number of buys: {self.buy_count}")
                print(f"Number of sells: {self.sell_count}")

                left_img = self.draw_coords_only(left_img, merged_left)
                right_img = self.draw_coords_only(right_img, merged_right)

                self.update_left.emit(left_img, merged_left)
                self.update_right.emit(right_img, merged_right)

                self.frame_count += 1
                print(f"\nFrame {self.frame_count} processed in {time.time() - start_time:.2f} sec.")

                time.sleep(0.001)
                
        except KeyboardInterrupt:
            print("KeyboardInterrupt caught, exiting...")
        finally:
            self.finished.emit()