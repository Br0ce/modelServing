FROM pytorch/torchserve:0.5.2-cpu

RUN mkdir config

CMD ["torchserve", "--start" ,"--ts-config config/config.properties"]
