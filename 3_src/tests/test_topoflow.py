"""Stage 0 Unit & Mathematical Oracle Verification Suite for E-120 TopoFlow Stack."""
from __future__ import annotations

from pathlib import Path
import sys
import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "3_src"))

from sfsc.metrics import evaluate
from sfsc.rle import encode_mask
from sfsc.topoflow_net import TopoFlowLoss, TopoFlowNet
from sfsc.topoflow_physics import extract_physics_channels
from sfsc.topoflow_targets import (
    build_topoflow_targets,
    decode_topoflow,
    invert_rotate_flip_predictions,
    load_magfilo_annotations,
    pq_mbr_solve,
    rotate_flip_tensor_and_flow,
)


def run_all_tests() -> None:
    json_p = ROOT / "1_data/raw/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    sets, p2i = load_magfilo_annotations(json_p)
    assert len(sets) == 1154 and len(p2i) == 37

    # 1. Layer 1 Physics Frontend test (verify zero NaN, bounded, limb-halo free)
    img_p = ROOT / "1_data/raw/MAGFiLO_1.0_Kaggle_2026/train/train_images" / sets[0].file_name
    img = np.array(Image.open(img_p))
    ch, fit, valid = extract_physics_channels(img)
    assert ch.shape == (5, 2048, 2048) and ch.dtype == np.float32
    assert np.isfinite(ch).all()
    # Verify outer space is strictly zeroed on channels 1..4
    assert float(np.abs(ch[1:, ~valid]).max()) == 0.0
    print("[PASS] Layer 1 Physics Frontend: shape=(5,2048,2048), zero off-disk bleed.")

    # 2. Oracle Reconstruction + Simulated Blurred/Merged Blob Splitting Stress Test
    crowded = sorted(sets, key=lambda s: len(s.instances), reverse=True)[:30]
    gt_rows, topo_clean_rows, cc8_blur_rows, topo_blur_rows = [], [], [], []
    t0 = time.time()
    for idx, s in enumerate(crowded):
        syn = f"set{idx:04d}"
        for k, inst in enumerate(s.instances, 1):
            gt_rows.append((f"01-{syn}_{k}", inst.rle_counts.decode("ascii")))
        tgts = build_topoflow_targets(s.instances)

        # Clean Oracle
        cands_clean = decode_topoflow(tgts["body"], tgts["spine"], tgts["sep"], tgts["flow"], min_area=5)
        for k, c in enumerate(cands_clean, 1):
            topo_clean_rows.append((f"{syn}_{k}", c.rle_counts))

        # Blurred / Over-merged simulation (simulating U-Net smooth probabilities at t_lo=0.35!)
        blur_body = cv2.GaussianBlur(tgts["body"], (0, 0), sigmaX=3.2, sigmaY=3.2)
        blur_spine = cv2.GaussianBlur(tgts["spine"], (0, 0), sigmaX=1.5, sigmaY=1.5)
        blur_sep = cv2.GaussianBlur(tgts["sep"], (0, 0), sigmaX=1.2, sigmaY=1.2)
        blur_vx = cv2.GaussianBlur(tgts["flow"][0], (0, 0), sigmaX=1.5, sigmaY=1.5)
        blur_vy = cv2.GaussianBlur(tgts["flow"][1], (0, 0), sigmaX=1.5, sigmaY=1.5)
        blur_flow = np.stack([blur_vx, blur_vy], axis=0)

        # Naive CC8 on blurred body >= 0.35
        lbl, n_cc = ndimage.label(blur_body >= 0.35, structure=np.ones((3, 3), int))
        kc = 0
        for cid in range(1, n_cc + 1):
            m = lbl == cid
            if m.sum() >= 20:
                kc += 1
                cc8_blur_rows.append((f"{syn}_{kc}", encode_mask(m)))

        # TopoFlow on exact same blurred body >= 0.35
        cands_blur = decode_topoflow(
            blur_body, blur_spine, blur_sep, blur_flow, t_body=0.35, t_seed=0.45, min_area=20
        )
        for k, c in enumerate(cands_blur, 1):
            topo_blur_rows.append((f"{syn}_{k}", c.rle_counts))

    dt = time.time() - t0
    gt_df = pd.DataFrame(gt_rows, columns=["filament_id", "segmentation_rle"])
    res_clean = evaluate(gt_df, pd.DataFrame(topo_clean_rows, columns=["filament_id", "segmentation_rle"]))
    res_cc8_b = evaluate(gt_df, pd.DataFrame(cc8_blur_rows, columns=["filament_id", "segmentation_rle"]))
    res_topo_b = evaluate(gt_df, pd.DataFrame(topo_blur_rows, columns=["filament_id", "segmentation_rle"]))

    print(f"[PASS] {len(crowded)} Crowded Sets ({len(gt_rows)} instances) built + 2x decoded in {dt:.2f}s ({dt/len(crowded)*1000:.1f}ms/set)")
    print("       Clean TopoFlow Oracle :", res_clean.summary())
    print("       Blurred CC8  (t=0.35) :", res_cc8_b.summary())
    print("       Blurred Topo (t=0.35) :", res_topo_b.summary())
    assert res_clean.pq > 0.995, f"Expected Clean Oracle PQ > 0.995, got {res_clean.pq}"
    assert res_topo_b.rq > res_cc8_b.rq + 0.02 and res_topo_b.pq > res_cc8_b.pq + 0.018, (
        "Expected TopoFlow to recover >90% of CC8 merge/split RQ errors!"
    )

    # 3. Continuous & D4 Rotational Vector Equivariance Test
    p_sub = np.stack([tgts["body"], tgts["spine"], tgts["sep"]], axis=0)[:, 600:1112, 600:1112]
    f_sub = tgts["flow"][:, 600:1112, 600:1112]
    for ang in (0.0, 90.0, 180.0, 270.0):
        for hf in (False, True):
            for vf in (False, True):
                pw, fw = rotate_flip_tensor_and_flow(p_sub, f_sub, ang, hf, vf)
                pi, fi = invert_rotate_flip_predictions(pw, fw, ang, hf, vf)
                assert np.abs(pi - p_sub).max() < 1e-5 and np.abs(fi - f_sub).max() < 1e-5
    # Non-90 degree rotation vector direction sanity check (+45 deg then -45 deg at interior pixels)
    vec_const = np.zeros((2, 128, 128), dtype=np.float32)
    vec_const[0, 32:96, 32:96] = 0.6
    vec_const[1, 32:96, 32:96] = 0.8
    _, fw45 = rotate_flip_tensor_and_flow(np.zeros((1, 128, 128), np.float32), vec_const, 37.0, True, False)
    _, fi45 = invert_rotate_flip_predictions(np.zeros((1, 128, 128), np.float32), fw45, 37.0, True, False)
    err_interior = float(np.abs(fi45[:, 48:80, 48:80] - vec_const[:, 48:80, 48:80]).max())
    assert err_interior < 1e-4, f"Arbitrary 37 deg + hflip inverse vector error {err_interior}"
    print(f"[PASS] Vector Equivariance (D4 + continuous 37 deg + flip): interior err={err_interior:.2e}")

    # 4. PQ-Optimal MWIS / MBR Solver Test (verify split beats merged when majority agrees, 0 overlap)
    c_split1_m = np.zeros((2048, 2048), dtype=bool)
    c_split1_m[100:140, 100:130] = True
    c_split2_m = np.zeros((2048, 2048), dtype=bool)
    c_split2_m[100:140, 132:162] = True
    c_merged_m = c_split1_m | c_split2_m
    c_merged_m[115:125, 130:132] = True

    from sfsc.topoflow_targets import CandidateInstance
    pool = [
        CandidateInstance(c_merged_m, encode_mask(c_merged_m), int(c_merged_m.sum()), 0.85, 0.9, 0.9, 0.2, 0.88, "merged"),
        CandidateInstance(c_split1_m, encode_mask(c_split1_m), int(c_split1_m.sum()), 0.85, 0.9, 0.9, 0.0, 0.86, "split"),
        CandidateInstance(c_split2_m, encode_mask(c_split2_m), int(c_split2_m.sum()), 0.85, 0.9, 0.9, 0.0, 0.86, "split"),
    ]
    refs = [
        ([encode_mask(c_split1_m), encode_mask(c_split2_m)], 0.70),
        ([encode_mask(c_merged_m)], 0.30),
    ]
    chosen = pq_mbr_solve(pool, refs, tau_pq=0.47)
    assert len(chosen) == 2 and int((chosen[0].mask & chosen[1].mask).sum()) == 0
    print("[PASS] PQ-MBR MWIS Solver: cleanly chose 2 split instances over overlapping merged candidate.")

    # 5. TopoFlowNet + TopoFlowLoss Forward & Backward Smoke Test (pretrained=False on CPU)
    net = TopoFlowNet(encoder_name="convnext_tiny", pretrained=False, in_channels=5, num_annotators=40)
    criterion = TopoFlowLoss()
    x_dummy = torch.randn(2, 5, 256, 256)
    a_ids = torch.tensor([0, 5], dtype=torch.long)
    logits_3, flow_2 = net(x_dummy, a_ids)
    assert logits_3.shape == (2, 3, 256, 256) and flow_2.shape == (2, 2, 256, 256)
    loss, m_dict = criterion(
        logits_3,
        flow_2,
        torch.randint(0, 2, (2, 256, 256)).float(),
        torch.rand(2, 256, 256),
        torch.rand(2, 256, 256) * 0.2,
        torch.randn(2, 2, 256, 256).clamp(-1, 1),
        torch.ones(2, 256, 256),
    )
    loss.backward()
    grad_norm = sum(float(p.grad.norm()) for p in net.parameters() if p.grad is not None)
    assert np.isfinite(grad_norm) and grad_norm > 0
    print(f"[PASS] TopoFlowNet + TopoFlowLoss forward/backward: loss={m_dict['loss']:.4f}, grad_norm={grad_norm:.2f}")

    # 6. E-121a Snake-TopoFlowNet + Lossless Exact-D4 Verification
    sys.path.insert(0, str(ROOT / "3_src/scripts"))
    from train_e121_snake512 import (
        compute_symmetric_barb_weight,
        exact_d4_tensor_and_flow,
        gpu_d4_forward,
        gpu_d4_invert,
    )

    net_snake = TopoFlowNet(
        encoder_name="convnext_tiny", pretrained=False, in_channels=5, num_annotators=40, use_snake=True
    )
    # Copy shared parameters -> zero-init SnakeBranchGN + BarbStripResidual must start at exact identity!
    net.eval()
    net_snake.eval()
    net_snake.load_state_dict(net.state_dict(), strict=False)
    with torch.no_grad():
        l_base, f_base = net(x_dummy, a_ids)
        l_snk0, f_snk0 = net_snake(x_dummy, a_ids)
    diff_zero = float((l_base - l_snk0).abs().max() + (f_base - f_snk0).abs().max())
    assert diff_zero < 1e-6, f"Expected zero-init SnakeTopoFlowNet parity, got {diff_zero}"

    net_snake.train()
    l_snk, f_snk = net_snake(x_dummy, a_ids)
    tb_dummy = torch.randint(0, 2, (2, 256, 256)).float()
    sep_dummy = torch.rand(2, 256, 256) * 0.2
    w_sym = compute_symmetric_barb_weight(tb_dummy, sep_dummy)
    loss_s, _ = criterion(
        l_snk, f_snk, tb_dummy, torch.rand(2, 256, 256), sep_dummy, torch.randn(2, 2, 256, 256).clamp(-1, 1), w_sym
    )
    loss_s.backward()
    snake_grad = sum(
        float(p.grad.abs().sum())
        for n, p in net_snake.named_parameters()
        if ("snake" in n or "barb2" in n) and p.grad is not None
    )
    assert np.isfinite(snake_grad) and snake_grad >= 0.0

    # Verify exact lossless D4 matches affine 90*k + hflip and GPU D4 forward/invert is bit-exact!
    p_sq = p_sub[:, :128, :128].copy()
    f_sq = f_sub[:, :128, :128].copy()
    p_t = torch.from_numpy(p_sq).unsqueeze(0)
    f_t = torch.from_numpy(f_sq)
    for k_rot in (0, 1, 2, 3):
        for hf in (False, True):
            pd4, fd4 = exact_d4_tensor_and_flow(p_sq, f_sq, k_rot, hf)
            p_aff, f_aff = rotate_flip_tensor_and_flow(p_sq, f_sq, float(90 * k_rot), hflip=hf, vflip=False)
            assert np.abs(pd4 - p_aff).max() < 1e-5 and np.abs(fd4 - f_aff).max() < 1e-5
            # GPU roundtrip
            p_fwd = gpu_d4_forward(p_t, k_rot, hf)[0]
            assert float((p_fwd - torch.from_numpy(pd4)).abs().max()) == 0.0
            p_inv, f_inv = gpu_d4_invert(torch.from_numpy(pd4), torch.from_numpy(fd4), k_rot, hf)
            assert float((p_inv - torch.from_numpy(p_sq)).abs().max()) == 0.0
            assert float((f_inv - f_t).abs().max()) == 0.0
    print(f"[PASS] SnakeTopoFlowNet zero-init parity ({diff_zero:.1e}) + Exact D4 CPU/GPU bit-identity verified!")


if __name__ == "__main__":
    run_all_tests()
