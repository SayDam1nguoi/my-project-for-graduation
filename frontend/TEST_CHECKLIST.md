# ✅ Test Checklist - Camera Integration

## Pre-requisites

- [ ] API đang chạy: `python api/main.py`
- [ ] Browser: Chrome hoặc Edge
- [ ] Camera và microphone có sẵn
- [ ] Internet không cần thiết (localhost)

## Test 1: Camera Access

### Steps:
1. [ ] Mở `frontend/app.html` trong browser
2. [ ] Click tab "📹 Camera Trực Tiếp"
3. [ ] Click button "📹 Bật Camera"
4. [ ] Browser hiển thị permission dialog
5. [ ] Click "Allow" để cho phép camera

### Expected Results:
- [ ] Video preview hiển thị (mirrored)
- [ ] Button "Bật Camera" disabled
- [ ] Button "Tắt Camera" enabled
- [ ] Button "Bắt Đầu Ghi" enabled
- [ ] Status hiển thị "Đang hoạt động" (màu xanh)
- [ ] Face Count cập nhật (0 hoặc 1)
- [ ] Emotion hiển thị emoji

### If Failed:
- Kiểm tra browser permissions
- Thử browser khác
- Kiểm tra camera không bị app khác dùng

---

## Test 2: Real-time Stats

### Steps:
1. [ ] Camera đã bật (từ Test 1)
2. [ ] Quan sát stats panel bên phải
3. [ ] Đợi 5-10 giây

### Expected Results:
- [ ] Face Count thay đổi (0 hoặc 1)
- [ ] Emotion thay đổi (😊, 😐, 😢, etc.)
- [ ] Stats cập nhật mỗi giây

### If Failed:
- Stats là simulated, nên sẽ random
- Nếu không thay đổi, check console logs

---

## Test 3: Recording Start

### Steps:
1. [ ] Camera đã bật
2. [ ] Click button "⏺ Bắt Đầu Ghi"

### Expected Results:
- [ ] Button "Bắt Đầu Ghi" ẩn
- [ ] Button "Dừng Ghi & Phân Tích" hiển thị
- [ ] Recording indicator hiển thị (đỏ, pulse)
- [ ] Timer hiển thị "00:00"
- [ ] Timer bắt đầu đếm (00:01, 00:02, ...)
- [ ] Status message: "⏺ Đang ghi hình..."

### If Failed:
- Check console for MediaRecorder errors
- Thử browser khác
- Check microphone permissions

---

## Test 4: Recording Timer

### Steps:
1. [ ] Recording đang chạy (từ Test 3)
2. [ ] Đợi 10 giây
3. [ ] Quan sát timer

### Expected Results:
- [ ] Timer đếm: 00:01, 00:02, ..., 00:10
- [ ] Format: MM:SS
- [ ] Cập nhật mỗi giây

### If Failed:
- Check JavaScript console
- Timer interval có thể bị clear

---

## Test 5: Recording Stop & Upload

### Steps:
1. [ ] Recording đang chạy
2. [ ] Nói vài câu vào microphone (10-30 giây)
3. [ ] Click button "⏹ Dừng Ghi & Phân Tích"

### Expected Results:
- [ ] Recording indicator ẩn
- [ ] Timer ẩn
- [ ] Button "Bắt Đầu Ghi" hiển thị lại
- [ ] Status: "⏹ Đã dừng ghi. Đang upload..."
- [ ] Status: "📤 Đang upload và phân tích video..."
- [ ] Không có error trong console

### If Failed:
- Check API đang chạy
- Check network tab trong DevTools
- Check CORS errors

---

## Test 6: Analysis & Results

### Steps:
1. [ ] Upload đã hoàn tất (từ Test 5)
2. [ ] Đợi 1-2 phút (API processing)

### Expected Results:
- [ ] Status: "✅ Phân tích hoàn tất!"
- [ ] Alert popup hiển thị
- [ ] Alert chứa:
  - [ ] Điểm Tổng (0-10)
  - [ ] Rating (XUẤT SẮC, TỐT, etc.)
  - [ ] Điểm Cảm xúc
  - [ ] Điểm Tập trung
  - [ ] Điểm Rõ ràng
  - [ ] Điểm Nội dung

### If Failed:
- Check API logs
- Check video format support
- Check Python dependencies

---

## Test 7: Stop Camera

### Steps:
1. [ ] Camera đang bật
2. [ ] Click button "⏹ Tắt Camera"

### Expected Results:
- [ ] Video preview dừng (màn hình đen)
- [ ] Button "Bật Camera" enabled
- [ ] Button "Tắt Camera" disabled
- [ ] Button "Bắt Đầu Ghi" disabled
- [ ] Status: "Chưa bật" (màu xám)
- [ ] Face Count: 0
- [ ] Emotion: -

### If Failed:
- Check stream cleanup
- Check video element srcObject

---

## Test 8: Multiple Recordings

### Steps:
1. [ ] Bật camera
2. [ ] Ghi video 1 (10 giây)
3. [ ] Dừng & phân tích
4. [ ] Đợi kết quả
5. [ ] Ghi video 2 (10 giây)
6. [ ] Dừng & phân tích
7. [ ] Đợi kết quả

### Expected Results:
- [ ] Cả 2 recordings thành công
- [ ] Cả 2 uploads thành công
- [ ] Cả 2 analyses thành công
- [ ] Không có memory leaks
- [ ] Không có errors

### If Failed:
- Check cleanup logic
- Check blob disposal
- Check API job management

---

## Test 9: Browser Compatibility

### Chrome
- [ ] Camera access: ✅
- [ ] Recording: ✅
- [ ] Upload: ✅
- [ ] Results: ✅

### Edge
- [ ] Camera access: ✅
- [ ] Recording: ✅
- [ ] Upload: ✅
- [ ] Results: ✅

### Firefox
- [ ] Camera access: ✅
- [ ] Recording: ✅
- [ ] Upload: ✅
- [ ] Results: ✅

### Safari (Optional)
- [ ] Camera access: ⚠️
- [ ] Recording: ⚠️
- [ ] Upload: ✅
- [ ] Results: ✅

---

## Test 10: Error Handling

### Test 10.1: API Not Running
1. [ ] Tắt API
2. [ ] Thử ghi và upload
3. [ ] Expected: Error message hiển thị

### Test 10.2: Camera Permission Denied
1. [ ] Deny camera permission
2. [ ] Expected: Error message hiển thị

### Test 10.3: Network Error
1. [ ] Disconnect network (sau khi bật camera)
2. [ ] Thử upload
3. [ ] Expected: Error message hiển thị

### Test 10.4: Large Video
1. [ ] Ghi video rất dài (>5 phút)
2. [ ] Thử upload
3. [ ] Expected: Có thể lỗi hoặc chậm

---

## Performance Tests

### Test 11: CPU Usage
- [ ] Bật camera
- [ ] Quan sát CPU usage
- [ ] Expected: <50% CPU

### Test 12: Memory Usage
- [ ] Bật camera
- [ ] Ghi 5 videos
- [ ] Quan sát memory
- [ ] Expected: Không tăng liên tục (no leaks)

### Test 13: Upload Speed
- [ ] Ghi video 30 giây
- [ ] Đo thời gian upload
- [ ] Expected: <5 giây

### Test 14: Processing Time
- [ ] Upload video 30 giây
- [ ] Đo thời gian phân tích
- [ ] Expected: 1-2 phút

---

## UI/UX Tests

### Test 15: Responsive Design
- [ ] Desktop (1920x1080): ✅
- [ ] Laptop (1366x768): ✅
- [ ] Tablet (768x1024): ✅
- [ ] Mobile (375x667): ⚠️ (camera có thể không tốt)

### Test 16: Dark Theme
- [ ] Background: Dark (#1a1a1a)
- [ ] Text: Light (#e0e0e0)
- [ ] Buttons: Gradient (purple-pink)
- [ ] Consistent với các tabs khác

### Test 17: Animations
- [ ] Recording indicator pulse: ✅
- [ ] Button hover effects: ✅
- [ ] Smooth transitions: ✅

---

## Integration Tests

### Test 18: Tab Switching
1. [ ] Bật camera trong tab Camera
2. [ ] Switch sang tab khác
3. [ ] Switch lại tab Camera
4. [ ] Expected: Camera vẫn hoạt động

### Test 19: Page Reload
1. [ ] Bật camera
2. [ ] Reload page
3. [ ] Expected: Camera tắt, cleanup OK

### Test 20: Multiple Tabs
1. [ ] Mở 2 tabs cùng app.html
2. [ ] Bật camera ở cả 2 tabs
3. [ ] Expected: Có thể lỗi (camera conflict)

---

## Final Checklist

### Code Quality
- [ ] No console errors
- [ ] No console warnings
- [ ] Clean code structure
- [ ] Comments where needed

### Documentation
- [ ] README updated
- [ ] API docs complete
- [ ] User guide available
- [ ] Troubleshooting guide

### Deployment Ready
- [ ] All features working
- [ ] Error handling complete
- [ ] Performance acceptable
- [ ] Browser compatibility tested

---

## Test Results Summary

**Date:** ___________
**Tester:** ___________
**Browser:** ___________
**OS:** ___________

**Total Tests:** 20
**Passed:** ___ / 20
**Failed:** ___ / 20
**Skipped:** ___ / 20

**Overall Status:** ✅ PASS / ❌ FAIL

**Notes:**
_________________________________
_________________________________
_________________________________

**Signature:** ___________
