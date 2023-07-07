import torch
from masks import ParticleMask, KinematicMask

# Validation loop
def validate(val_loader, tae, classifier, device, criterion, class_criterion, mask, epoch, num_epochs, val_loss_min, val_loss_min_2, save_path):
    # Validation loop
    tae.eval()  # Set the tae to evaluation mode
    classifier.eval()
    val_losses = []
    val_losses_2 = []
    with torch.no_grad():  # Disable gradient calculations
        for val_batch in val_loader:
            # Move the data to the device
            val_inputs, val_labels = val_batch
            val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device)
            if mask is not None:
                if mask == 0:
                    mask_layer = ParticleMask(4)
                else:
                    mask_layer = KinematicMask(mask)
                # Mask input data
                masked_val_inputs = mask_layer(val_inputs)

            if (val_labels == 1).any():
              trimmed_masked_val_inputs = masked_val_inputs[val_labels == 1]
              trimmed_val_outputs = tae(trimmed_masked_val_inputs, trimmed_masked_val_inputs)
              trimmed_val_outputs = torch.reshape(trimmed_val_outputs, (trimmed_val_outputs.size(0),
                                                                        trimmed_val_outputs.size(1) * trimmed_val_outputs.size(2)))
              trimmed_val_inputs = val_inputs[val_labels == 1]
              trimmed_val_inputs = torch.reshape(trimmed_val_inputs, (trimmed_val_inputs.size(0),
                                                                      trimmed_val_inputs.size(1) * trimmed_val_inputs.size(2)))
              val_loss = criterion(trimmed_val_outputs, trimmed_val_inputs, zero_padded=[3,5,7])
              val_losses.append(val_loss.item())
            else:
              val_losses.append(0)

            # Forward pass
            val_outputs = tae(masked_val_inputs, masked_val_inputs)

            # Reshape tensors
            val_outputs = torch.reshape(val_outputs, (val_outputs.size(0),
                                                      val_outputs.size(1) * val_outputs.size(2)))
            masked_val_inputs = torch.reshape(masked_val_inputs, (masked_val_inputs.size(0),
                                                                  masked_val_inputs.size(1) * masked_val_inputs.size(2)))

            # Forward pass for classifier
            val_outputs_2 = classifier(torch.cat((val_outputs, masked_val_inputs), axis=1)).squeeze(1)

            val_loss_2 = class_criterion(val_outputs_2, val_labels.float())
            val_losses_2.append(val_loss_2.item())

    val_loss_mean = sum(val_losses) / len(val_losses)
    val_loss_mean_2 = sum(val_losses_2) / len(val_losses_2)

    # Print total loss for the epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss_mean:.4f}, Val Class Loss: {val_loss_mean_2:.4f}")

    # Save files if better than best performance
    if val_loss_mean < val_loss_min:
        val_loss_min = val_loss_mean
        torch.save(tae.state_dict(), save_path + '/TAE_best_' + model_name)
    if val_loss_mean_2 < val_loss_min_2:
        val_loss_min_2 = val_loss_mean_2
        torch.save(classifier.state_dict(), save_path + '/Classifier_best_' + model_name)
    return val_loss_min, val_loss_min_2