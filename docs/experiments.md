## 2021-04-07

Worked on the Semeval dataset. Took the Entity-Destination class to build a one-versus-rest classifier. 

The labels in the dataset:
- Cause-Effect(e1,e2): 344
- Cause-Effect(e2,e1): 659
- Component-Whole(e1,e2): 470
- Component-Whole(e2,e1): 471
- Content-Container(e1,e2): 374
- Content-Container(e2,e1): 166
- Entity-Destination(e1,e2): 844
- Entity-Destination(e2,e1): 1
- Entity-Origin(e1,e2): 568
- Entity-Origin(e2,e1): 148
- Instrument-Agency(e1,e2): 97
- Instrument-Agency(e2,e1): 407
- Member-Collection(e1,e2): 78
- Member-Collection(e2,e1): 612
- Message-Topic(e1,e2): 490
- Message-Topic(e2,e1): 144
- Other: 1410
- Product-Producer(e1,e2): 323
- Product-Producer(e2,e1): 394

We selected a class with the most samples: Entity-Destination(e1,e2). The distribution of the classes after this:

__1__ - 844 

__0__ - 7156

Then we generated all the subgraphs for the sentences up to three edges. The vocabulary size was 499095. We selected the most frequent 7000 features from here and trained various white box classifiers. Logistic regression performed the best.


|           |       |
| --------- | ----- |
| **Precision** | 0.8046875 |
| **Recall**    | 0.7953667  |
| **Fscore**          |    0.8   |

Then we selected the best features from the model and converted them into rules. The top three rules:
|           |
| --------- |
| **'(u_29 / to  :2 (u_4 / entity2))'** |
| **'(u_26 / into  :2 (u_4 / entity2))'** |
| **'(u_82 / place  :2 (u_2 / entity1))'** | 

When we run the GraphMatcher algorithm we get the following results:
|           |       |
| --------- | ----- |
| **Precision** | 0.71111 |
| **Recall**    | 0.74131 |
| **Fscore**          |    0.72589   |