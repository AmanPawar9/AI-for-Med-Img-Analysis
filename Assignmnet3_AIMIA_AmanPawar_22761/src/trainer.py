import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def run(generator, discriminator, generator_optimizer, discriminator_optimizer, 
            criterion, train_loader, val_loader, num_epochs, device, latent_size):
    
    if not os.path.exists("./ModelWeights"):
        os.makedirs("./ModelWeights")
    if not os.path.exists("./Results"):
        os.makedirs("./Results")


    best_val_loss = float('inf')
    best_generator_weights = None
    best_discriminator_weights = None

    # Training loop
    train_losses = []
    val_losses = []
    train_accuracy = []
    val_accuracy = []

    for epoch in range(num_epochs):
        train_loss_total = 0.0
        val_loss_total = 0.0
        train_correct = 0
        val_correct = 0
        train_total = 0
        val_total = 0
        
        # Training
        generator.train()
        discriminator.train()
        generator_updates = 0

        train_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} - Training')
        for i, (real_images, _) in enumerate(train_bar):
            real_labels = torch.ones(real_images.size(0), 1).to(device)
            fake_labels = torch.zeros(real_images.size(0), 1).to(device)
            real_images = real_images.to(device)

            
            # Train Discriminator
            discriminator_optimizer.zero_grad()
            real_outputs = discriminator(real_images)
            d_loss_real = criterion(real_outputs, real_labels)
            d_loss_real.backward()

            noise = torch.randn(real_images.size(0), latent_size, 1, 1).to(device)
            fake_images = generator(noise)
            fake_outputs = discriminator(fake_images.detach())
            d_loss_fake = criterion(fake_outputs, fake_labels)
            d_loss_fake.backward()

            d_loss = d_loss_real + d_loss_fake
            
            if i%3==0: # Training the discriminator only once in every 3 epochs to get consistancy for nash eqm
                discriminator_optimizer.step()

            # Train Generator
            generator_optimizer.zero_grad()
            fake_images = generator(noise)
            outputs = discriminator(fake_images)
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()
            generator_optimizer.step()

            train_loss_total += d_loss + g_loss

            # Calculate accuracy
            train_total += real_labels.size(0) * 2
            train_correct += ((real_outputs >= 0.5).sum().item() + (fake_outputs < 0.5).sum().item())

            train_bar.set_postfix(
                {'Generator Loss': g_loss.item(), 'Discriminator Loss': d_loss.item()}
            )

        train_loss_avg = train_loss_total / len(train_loader)
        train_losses.append(train_loss_avg)
        train_accuracy.append(train_correct / train_total)

        # Validation
        generator.eval()
        discriminator.eval()
        val_bar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} - Validation')
        with torch.no_grad():
            for i, (real_images, _) in enumerate(val_bar):
                real_labels = torch.ones(real_images.size(0), 1).to(device)
                fake_labels = torch.zeros(real_images.size(0), 1).to(device)
                real_images = real_images.to(device)

                val_outputs_real = discriminator(real_images)
                val_d_loss_real = criterion(val_outputs_real, real_labels)

                noise = torch.randn(real_images.size(0), latent_size, 1, 1).to(device)
                fake_images = generator(noise)
                val_outputs_fake = discriminator(fake_images.detach())
                val_d_loss_fake = criterion(val_outputs_fake, fake_labels)

                val_d_loss = val_d_loss_real + val_d_loss_fake
                val_loss_total += val_d_loss

                # Calculate accuracy
                val_total += real_labels.size(0) * 2
                val_correct += ((val_outputs_real >= 0.5).sum().item() + (val_outputs_fake < 0.5).sum().item())

                val_bar.set_postfix({'Validation Loss': val_d_loss.item()})

            val_loss_avg = val_loss_total / len(val_loader)
            val_losses.append(val_loss_avg)
            val_accuracy.append(val_correct / val_total)

            # Save best model weights based on validation loss
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                best_generator_weights = generator.state_dict()
                best_discriminator_weights = discriminator.state_dict()

    # Save best model weights to files
    torch.save(best_generator_weights, './ModelWeights/best_generator.pth')
    torch.save(best_discriminator_weights, './ModelWeights/best_discriminator.pth')

    # Save losses and accuracy to a text file
    with open('Results/GANtraining_results.txt', 'w') as file:
        file.write('Training Losses\tValidation Losses\tTraining Accuracy\tValidation Accuracy')
        for train_loss, val_loss, train_acc, val_acc in zip(train_losses,val_losses,train_accuracy,val_accuracy):
            file.write(f'{train_loss}\t{val_loss}\t{train_acc}\t{val_acc}\n')



# Define loss function and optimizer
def loss_function(reconstructed_x, x, mu, log_var, device):
    # Reconstruction loss
    reconstruction_loss = nn.functional.cross_entropy(reconstructed_x, x, reduction='sum').to(device)

    # KL divergence regularization
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()).to(device)

    return reconstruction_loss + kl_divergence

# Define function to train the VAE model
def train_vae(model, criterion, optimizer, train_loader, val_loader, device, num_epochs=10):

    if not os.path.exists("./ModelWeights"):
        os.makedirs("./ModelWeights")
    if not os.path.exists("./Results"):
        os.makedirs("./Results")

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    model.to(device)
    

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training')
        for data in train_bar:
            inputs, _ = data
            inputs = inputs.to(device)

            optimizer.zero_grad()
            reconstructed, mu, log_var = model(inputs)
            reconstructed = reconstructed.view(-1, 3, 28, 28)
            loss = criterion(reconstructed, inputs, mu, log_var, device)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_bar.set_postfix({'Training Loss': loss.item()})

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # print(f'Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.4f}')

        # Validation
        model.eval()
        val_loss = 0.0
        val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation')
        with torch.no_grad():
            for data in val_bar:
                inputs, _ = data
                inputs = inputs.to(device)

                reconstructed, mu, log_var = model(inputs)
                reconstructed = reconstructed.view(-1, 3, 28, 28)
                loss = criterion(reconstructed, inputs, mu, log_var, device)

                val_loss += loss.item()
                val_bar.set_postfix({'Validation Loss': loss.item()})
            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)

            # print(f'Epoch [{epoch + 1}/{num_epochs}] - Validation Loss: {val_loss:.4f}')

            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), './ModelWeights/best_vae_model.pth')

    # Save losses to a text file
    with open('./Results/VAElosses.txt', 'w') as file:
        file.write('Train Losses\tValidation Losses\n')
        for train_loss, val_loss in zip(train_losses, val_losses):
            file.write(f'{train_loss}\t{val_loss}\n')