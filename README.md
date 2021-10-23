# Domain Adaptation with Invariant RepresentationLearning: What Transformations to Learn?


Repository Structure:

>DSAN<br>
|└───amazon<br>
|&emsp;&emsp;&emsp;&emsp;└── dataset (Amazon dataset)<br>
|&emsp;&emsp;&emsp;&emsp;├── result<br>
|&emsp;&emsp;&emsp;&emsp;├── amazon_utils.py<br>
|&emsp;&emsp;&emsp;&emsp;├── dsan.py<br>
|&emsp;&emsp;&emsp;&emsp;└── flip_gradient.py<br>
|&emsp;&emsp;&emsp;&emsp;└── logger.py<br>
|────imageclef<br>
|&emsp;&emsp;&emsp;&emsp;└── dataset (ImageCLEF dataset)<br>
|&emsp;&emsp;&emsp;&emsp;├── logs<br>
|&emsp;&emsp;&emsp;&emsp;├── utils.py<br>
|&emsp;&emsp;&emsp;&emsp;├── dsan.py<br>
|&emsp;&emsp;&emsp;&emsp;└── flip_gradient.py<br>
>
Instructions on running the code: 
##1. Run the following command<br>
```   


# for Amazon
cd amazon
python dsan.py --src $source_domain_name --tgt $target_domain_name 

# for ImageCLEF
cd imageclef
python dsan.py --src $source_domain_name --tgt $target_domain_name
```

##2. Compute environment for our experiments:<br>
CPU: Intel 7700k<br>
GPU: GeForce RTX2070<br>
32 GB Memory<br>
##3. Table of the experiment result for Amazon:<br>

Model|B&rarr;D|B&rarr;E|B&rarr;K|D&rarr;B|D&rarr;E|D&rarr;K|E&rarr;D|E&rarr;B|E&rarr;K|K&rarr;B|K&rarr;D|K&rarr;E|
----  | ---      | ---      | ---      | ---      | ---      | ---      | ---      | ---      | ---      | ---      | ---      | ---      |
DIRT-T   | 78.6     | 76.1     |  75.5    |  76.8    |  75.2    |  79.1    |  69.6    |  71.0    |  84.2    | 69.2     |  73.3    |  79.5    |
MDD     | 77.1      | 74.4    |   77.0   | 74.7     |   74.1   |  76.3    |  72.4    | 70.2     |  83.3    |69.3      |73.2      |   82.8   |
DSAN    | 82.7     | 80.8     |  82.6    |  79.5    |  81.4    |  85.3    |  76.7    |  75.1    |  88.0    | 73.8     |  77.3    |  85.0    |

##4. Table of the experiment result for ImageCLEF:<br>

Model|i&rarr;p|p&rarr;i|i&rarr;c|c&rarr;i|c&rarr;p|p&rarr;c|
----  | ---   | ---    | ---    | ---    | ---    | ---    |
CDAN  | 77.7  | 90.7   |  97.7  | 91.3   |  74.2  | 94.3   | 
SPL   | 78.8  | 94.5   |  96.7  | 95.7   |  80.5  | 96.3   |  
Ours  | 81.0  | 95.7   |  97.3  | 95.3   |  80.5  | 97.0   | 
