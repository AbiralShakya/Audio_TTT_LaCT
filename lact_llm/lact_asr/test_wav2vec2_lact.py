#!/usr/bin/env python3
"""
Test script for Wav2Vec2 with LaCT integration
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from modeling_asr_wav2vec2_lact import Wav2Vec2WithLaCT, BaselineWav2Vec2

def test_wav2vec2_lact_integration():
    """Test the Wav2Vec2 with LaCT integration"""
    
    print("Testing Wav2Vec2 with LaCT Integration")
    print("=" * 50)
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Load Wav2Vec2 model
        print("1. Loading Wav2Vec2 model...")
        model_id = "facebook/wav2vec2-base-960h"
        processor = Wav2Vec2Processor.from_pretrained(model_id)
        base_model = Wav2Vec2ForCTC.from_pretrained(model_id)
        base_model.eval()
        print(f"   ✓ Loaded {model_id}")
        
        # Create baseline model
        print("2. Creating baseline model...")
        baseline = BaselineWav2Vec2(base_model)
        baseline.to(device)
        baseline.eval()
        print("   ✓ Baseline model created")
        
        # Create LaCT model
        print("3. Creating Wav2Vec2 with LaCT model...")
        lact_model = Wav2Vec2WithLaCT(
            wav2vec2_model=base_model,
            num_lact_layers=2,
            lact_hidden_size=768,
            num_lact_heads=4,
            lact_chunk_size=512,
            lact_lr_scale=0.01,
            use_momentum=True
        )
        lact_model.to(device)
        lact_model.eval()
        print("   ✓ LaCT model created")
        
        # Create dummy audio input
        print("4. Creating dummy audio input...")
        batch_size = 1
        sequence_length = 16000  # 1 second at 16kHz
        dummy_audio = torch.randn(batch_size, sequence_length)
        
        # Process with Wav2Vec2 processor
        dummy_audio_np = dummy_audio.squeeze().numpy()
        inputs = processor(
            dummy_audio_np,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        input_values = inputs["input_values"].to(device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        print(f"   ✓ Input shape: {input_values.shape}")
        
        # Test baseline model
        print("5. Testing baseline model...")
        with torch.no_grad():
            baseline_outputs = baseline(input_values, attention_mask=attention_mask)
            baseline_logits = baseline_outputs.logits if hasattr(baseline_outputs, 'logits') else baseline_outputs['logits']
            print(f"   ✓ Baseline output shape: {baseline_logits.shape}")
        
        # Test LaCT model
        print("6. Testing LaCT model...")
        lact_model.reset_fast_weights()  # Reset fast weights
        
        with torch.no_grad():
            lact_outputs = lact_model(input_values, attention_mask=attention_mask)
            lact_logits = lact_outputs['logits']
            print(f"   ✓ LaCT output shape: {lact_logits.shape}")
        
        # Verify output shapes match
        assert baseline_logits.shape == lact_logits.shape, f"Shape mismatch: {baseline_logits.shape} vs {lact_logits.shape}"
        print("   ✓ Output shapes match")
        
        # Test decoding
        print("7. Testing decoding...")
        baseline_pred_ids = torch.argmax(baseline_logits, dim=-1)
        lact_pred_ids = torch.argmax(lact_logits, dim=-1)
        
        baseline_text = processor.batch_decode(baseline_pred_ids)[0]
        lact_text = processor.batch_decode(lact_pred_ids)[0]
        
        print(f"   ✓ Baseline decoded: {baseline_text[:50]}...")
        print(f"   ✓ LaCT decoded: {lact_text[:50]}...")
        
        # Test multiple forward passes (simulate chunk processing)
        print("8. Testing multiple forward passes...")
        for i in range(3):
            lact_model.reset_fast_weights()
            with torch.no_grad():
                outputs = lact_model(input_values, attention_mask=attention_mask)
                print(f"   ✓ Pass {i+1}: {outputs['logits'].shape}")
        
        print("\n" + "=" * 50)
        print("✓ All tests passed! Wav2Vec2 with LaCT integration is working.")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_lact_layer_components():
    """Test individual LaCT layer components"""
    
    print("\nTesting LaCT Layer Components")
    print("=" * 50)
    
    try:
        from modeling_asr_wav2vec2_lact import LaCTLayer
        
        # Create LaCT layer
        lact_layer = LaCTLayer(
            hidden_size=768,
            num_lact_heads=4,
            chunk_size=512,
            lr_scale=0.01,
            use_momentum=True
        )
        
        # Test forward pass
        batch_size = 2
        seq_length = 1024
        hidden_size = 768
        
        x = torch.randn(batch_size, seq_length, hidden_size)
        output = lact_layer(x)
        
        print(f"✓ Input shape: {x.shape}")
        print(f"✓ Output shape: {output.shape}")
        print(f"✓ Output matches input shape: {output.shape == x.shape}")
        
        # Test fast weight updates
        print("✓ Fast weight matrices initialized:")
        print(f"  - w0 shape: {lact_layer.w0.shape}")
        print(f"  - w1 shape: {lact_layer.w1.shape}")
        print(f"  - w2 shape: {lact_layer.w2.shape}")
        
        if lact_layer.use_momentum:
            print("✓ Momentum buffers initialized:")
            print(f"  - w0_momentum shape: {lact_layer.w0_momentum.shape}")
            print(f"  - w1_momentum shape: {lact_layer.w1_momentum.shape}")
            print(f"  - w2_momentum shape: {lact_layer.w2_momentum.shape}")
        
        print("✓ LaCT layer components working correctly")
        return True
        
    except Exception as e:
        print(f"❌ LaCT layer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Wav2Vec2 with LaCT Integration Test Suite")
    print("=" * 60)
    
    # Run tests
    test1_passed = test_wav2vec2_lact_integration()
    test2_passed = test_lact_layer_components()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Integration Test: {'✓ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Component Test:  {'✓ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Ready to run experiments.")
        print("\nNext steps:")
        print("1. Run: python run_asr_experiment_updated.py")
        print("2. Plot: python plot_wav2vec2_lact_results.py")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.") 