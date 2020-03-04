import numpy
import torch


def image_to_text(contents, captions, images, npts=None, verbose=False):
    """
    Images->Text (Image Annotation)
    Images: (N, K) matrix of images
    Captions: (N, K) matrix of captions
    """
    if npts == None:
        npts = images.size()[0]
        npts = int(npts)

    ranks = numpy.zeros(npts)
    gen = []
    for index in range(npts):
        # Get query image
        im = images[index].unsqueeze(0)

        # Compute scores
        d = torch.mm(im, captions.t())
        d_sorted, inds = torch.sort(d, descending=True)
        inds = inds.data.squeeze(0).cpu().numpy()

        # Score
        ranks[index] = numpy.where(inds == index)[0][0]

        # gen[index] = contents[inds[0]].split("_")[-1]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    # with open('err_im_to_text', 'w') as f:
    #     f.write('>10:' + str([contents[i] for i in numpy.where(ranks > 10)[0]]) + "\n")
    #     f.write('>5:' + str([contents[i] for i in numpy.where(ranks > 5)[0]]) + "\n")
    #     f.write('>1:' + str([contents[i] for i in numpy.where(ranks > 1)[0]]) + "\n")
    #     f.write(gen)

    if verbose:
        print("		* Image to text scores: R@1: %.1f, R@5: %.1f, R@10: %.1f, Medr: %.1f" % (r1, r5, r10, medr))
    return r1 + r5 + r10, (r1, r5, r10, medr)


def text_to_image(contents, captions, images, npts=None, verbose=False):
    if npts == None:
        npts = images.size()[0]
        npts = int(npts)

    ims = torch.cat([images[i].unsqueeze(0) for i in range(0, len(images), 1)])

    ranks = numpy.zeros(npts)
    for index in range(npts):

        # Get query captions
        queries = captions[index]
        queries = queries.unsqueeze(0)

        # Compute scores
        d = torch.mm(queries, ims.t())
        for i in range(d.size()[0]):
            d_sorted, inds = torch.sort(d[i], descending=True)
            inds = inds.data.squeeze(0).cpu().numpy()
            ranks[1 * index + i] = numpy.where(inds == index)[0][0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    # with open('err_txt_to_img', 'w') as f:
    #     f.write('>10:' + str([contents[i] for i in numpy.where(ranks > 10)[0]]) + "\n")
    #     f.write('>5:' + str([contents[i] for i in numpy.where(ranks > 5)[0]]) + "\n")
    #     f.write('>1:' + str([contents[i] for i in numpy.where(ranks > 1)[0]]) + "\n")
    if verbose:
        print("		* Text to image scores: R@1: %.1f, R@5: %.1f, R@10: %.1f, Medr: %.1f" % (r1, r5, r10, medr))
    return r1 + r5 + r10, (r1, r5, r10, medr)


