import torch
from validate import validate
from masks import ParticleMask, KinematicMask
from argparse import ArgumentParser

def train(train_loader, val_loader, tae, classifier, device, optimizer, optimizer_2, criterion, class_criterion, mask=None, num_epochs:int=50, val_loss_min:int=999, val_loss_min_2:int=999, save_path:str='./saved_models', model_name:str=''):
    # Create an outputs folder to store config files
    try:
        os.mkdir('./outputs/' + model_name)
    except:
        print('./outputs/' + model_name + ' Already Exists')
        
    for epoch in range(num_epochs):
        tae.train()
        classifier.train()
        running_loss = 0.0
        running_loss_2 = 0.0
        for batch_idx, batch in enumerate(train_loader):
            # Move the data to the device
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Apply particle mask
            if mask is not None:
                if mask == 0:
                    mask_layer = ParticleMask(4)
                else:
                    mask_layer = KinematicMask(mask)
                # Mask input data
                masked_inputs = mask_layer(inputs)

            # Zero the gradients
            optimizer_2.zero_grad()

            # Forward pass
            outputs = tae(masked_inputs, masked_inputs)

            # Reshape tensors
            outputs = torch.reshape(outputs, (outputs.size(0),
                                              outputs.size(1) * outputs.size(2)))
            flat_masked_inputs = torch.reshape(masked_inputs, (masked_inputs.size(0),
                                                               masked_inputs.size(1) * masked_inputs.size(2)))

            # Forward pass for classifier
            outputs_2 = classifier(torch.cat((outputs, flat_masked_inputs), axis=1)).squeeze(1)

            # Caclulate the loss
            loss_2 = class_criterion(outputs_2, labels.float())

            # Backward pass
            loss_2.backward()

            # Update the parameters
            optimizer_2.step()

            # Zero the gradients
            optimizer.zero_grad()

            if (labels == 1).any():
              trimmed_masked_inputs = masked_inputs[labels == 1]
              trimmed_outputs = tae(trimmed_masked_inputs, trimmed_masked_inputs)
              trimmed_outputs = torch.reshape(trimmed_outputs, (trimmed_outputs.size(0),
                                                                trimmed_outputs.size(1) * trimmed_outputs.size(2)))
              trimmed_inputs = inputs[labels == 1]
              trimmed_inputs = torch.reshape(trimmed_inputs, (trimmed_inputs.size(0),
                                                              trimmed_inputs.size(1) * trimmed_inputs.size(2)))
              loss = criterion(trimmed_outputs, trimmed_inputs, zero_padded=[3,5,7])
            else:
              loss = torch.zeros(1)

            # Calculate the loss

            # Backward pass
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Update running loss
            running_loss += loss.item()
            running_loss_2 += loss_2.item()

            # Print running loss every 500 batches
            if (batch_idx + 1) % 500 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {running_loss / 500:.4f}, Class Loss: {running_loss_2 / 500:.4f}")
                running_loss = 0.0
                running_loss_2 = 0.0

        val_loss_min, val_loss_min_2 = validate(val_loader, tae, classifier, criterion, class_criterion, mask, epoch, num_epochs, val_loss_min, val_loss_min_2, save_path, model_name)
    return val_loss_min, val_loss_min_2
