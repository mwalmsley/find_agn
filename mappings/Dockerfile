FROM ubuntu

RUN apt-get update
RUN apt-get install -y git
RUN apt-get install -y gfortran gdb make
RUN apt-get install wget unzip

# into folder mappings
RUN git clone https://bitbucket.org/RalphSutherland/mappings.git  

# into root
RUN wget --no-check-certificate -O=atmos.zip https://miocene.anu.edu.au/mappings/MV_atmos-5011.zip  
# makes atmos folder in lab
# place in d directory, (u) files newer or not already there
RUN unzip -d /mappings/lab -u *.zip   

WORKDIR mappings/src
RUN make build
RUN make install
WORKDIR ../lab

CMD ./map51