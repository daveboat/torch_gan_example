import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from model import Generator, Discriminator
from losses import generator_loss, discriminator_loss

def train_one_epoch(epoch, dataloader, generator, discriminator, generator_optimizer, discriminator_optimizer,
                    generator_loss, discriminator_loss, device, noise_dim=100, log_frequency=20):
    generator.train()
    discriminator.train()

    print('Starting epoch {}'.format(epoch))

    for batch_idx, (images, target) in enumerate(dataloader):
        images = images.to(device)

        # fetch batch size
        batch_len = images.size(0)

        # create gaussian noise vector as input to the generator
        noise = torch.randn((batch_len, noise_dim)).to(device)

        # train discriminator by running the random noise through the generator for the fake images, and taking a batch
        # of real images from the dataloader. The discriminator classifies both of them, then the loss is minimized by
        # classifying both correctly
        discriminator_optimizer.zero_grad()
        generated_images = generator(noise)
        real_output = discriminator(images)
        fake_output = discriminator(generated_images)

        disc_loss = discriminator_loss(real_output, fake_output)
        disc_loss.backward()
        discriminator_optimizer.step()

        # train generator by running the random noise through the generator and the discriminator. Have to do this again
        # because torch clears graphs for memory reasons (?) after backward() is finished, so can't backpropogate again
        # without retain_graph=True, unless another forward pass needs to be done, which is what we do here
        generator_optimizer.zero_grad()
        generated_images = generator(noise)
        fake_output = discriminator(generated_images)

        gen_loss = generator_loss(fake_output)
        gen_loss.backward()
        generator_optimizer.step()

        # print some info for this batch
        if batch_idx % log_frequency == 0:
            print('Batch {}/{}: generator loss = {:.3f}, discriminator loss = {:.3f}'.format(batch_idx, len(dataloader),
                                                                                           gen_loss.item(),
                                                                                             disc_loss.item()))


if __name__ == '__main__':
    # training hyperparameters
    epochs = 50
    batch_size = 256
    noise_dim = 100
    seed = 42
    lr = 1e-4

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(seed)

    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])

    # create dataloader
    # datasets are tuples of (28x28 PIL image, target as a number)
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)

    # initialize models
    generator = Generator(noise_dim=noise_dim)
    discriminator = Discriminator()

    generator.to(device)
    discriminator.to(device)

    # create optimizers for the generator and discriminator
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)

    # initialize losses
    generator_criterion = generator_loss
    discriminator_criterion = discriminator_loss

    for i in range(epochs):
        train_one_epoch(i, dataloader, generator, discriminator, generator_optimizer, discriminator_optimizer,
                        generator_criterion, discriminator_criterion, device)

    torch.save(generator.state_dict(), 'generator.pth')
