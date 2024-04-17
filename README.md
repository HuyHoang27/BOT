# Bot Hỗ Trợ Bán Hàng

Bot Hỗ Trợ Bán Hàng là một ứng dụng AI được phát triển để tương tác với khách hàng và xử lý thanh toán hóa đơn. Ứng dụng này sử dụng các thư viện Python để hoạt động. Hãy làm theo các bước dưới đây để cài đặt và chạy bot.

## Cài Đặt

1. Tạo Môi Trường Ảo:

    ```bash
    virtualenv env
    ```

2. Kích Hoạt Môi Trường Ảo:
    ```bash
    env\Scripts\activate
    ```

3. Cài Đặt Thư Viện: Cài đặt các thư viện cần thiết từ tệp requirements.txt bằng lệnh sau:

    ```bash
    pip install -r requirements.txt
    ```

4. Thoát khỏi môi trường ảo khi bạn đã hoàn thành:

    ```bash
    deactivate
    ```

## Chức Năng

Bot Hỗ Trợ Bán Hàng hiện có hai chức năng chính:

1. Tương Tác với Khách Hàng: Bot có khả năng tương tác với khách hàng, trả lời câu hỏi, cung cấp thông tin sản phẩm và giúp họ đặt hàng.
2. Thanh Toán Hóa Đơn: Bot cũng có thể xử lý thanh toán hóa đơn từ khách hàng.

## Sử Dụng
1. Để sử dụng bot, hãy chạy mã bằng lệnh sau:
    ```bash
    python main.py
    ```

2. Sau đó, bạn có thể tương tác với bot theo các phím tắt sau:
    * Phím Space: Bắt đầu ghi âm và kết thúc ghi âm.
    * Phím T: Thực hiện thanh toán.
    * Phím Q: Kết thúc trò chuyện với bot.