from torch.nn import functional as F

def cal_feature_padding_size(p2_shape):
        ps_list = [64, 32, 16, 8]
        ori_size = []
        paddings = []
        unpaddings = []
        padded_size = []

        ori_size.append(p2_shape)
        for i in range(len(ps_list) - 1):
            h, w = ori_size[-1]
            ori_size.append(((h + 1) // 2, (w + 1) // 2))

        for i, ps in enumerate(ps_list):
            h = ori_size[i][0]
            w = ori_size[i][1]

            h_pad_len = ps - h % ps if h % ps != 0 else 0
            w_pad_len = ps - w % ps if w % ps != 0 else 0

            paddings.append(
                (
                    w_pad_len // 2,
                    w_pad_len - w_pad_len // 2,
                    h_pad_len // 2,
                    h_pad_len - h_pad_len // 2,
                )
            )
            unpaddings.append(
                (
                    0 - (w_pad_len // 2),
                    0 - (w_pad_len - w_pad_len // 2),
                    0 - (h_pad_len // 2),
                    0 - (h_pad_len - h_pad_len // 2),
                )
            )

        for i, p in enumerate(paddings):
            h = ori_size[i][0]
            w = ori_size[i][1]
            h_pad_len = p[2] + p[3]
            w_pad_len = p[0] + p[1]
            padded_size.append((h + h_pad_len, w + w_pad_len))

        return {
            "ori_size": ori_size,
            "paddings": paddings,
            "unpaddings": unpaddings,
            "padded_size": padded_size,
        }

def feature_padding( features, pad_info):
    p2, p3, p4, p5 = features
    paddings = pad_info["paddings"]

    p2 = F.pad(p2, paddings[0], mode="reflect")
    p3 = F.pad(p3, paddings[1], mode="reflect")
    p4 = F.pad(p4, paddings[2], mode="reflect")
    p5 = F.pad(p5, paddings[3], mode="reflect")
    return [p2, p3, p4, p5]

def feature_unpadding( features, pad_info):
    p2, p3, p4, p5 = features
    unpaddings = pad_info["unpaddings"]

    p2 = F.pad(p2, unpaddings[0])
    p3 = F.pad(p3, unpaddings[1])
    p4 = F.pad(p4, unpaddings[2])
    p5 = F.pad(p5, unpaddings[3])
    return [p2, p3, p4, p5]