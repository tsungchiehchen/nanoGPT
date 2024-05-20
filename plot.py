import matplotlib.pyplot as plt

def clean_loss(loss):
    # split the string by comma and remove the parentheses
    loss = loss.split(',')
    loss = [float(loss.replace('tensor(', '').replace(')', '')) for loss in loss]
    return loss

iteration_numbers = list(range(0, 5250, 250))

# Question 1
q1_val_loss = "tensor(4.2823), tensor(2.0708), tensor(1.7248), tensor(1.5904), tensor(1.5201), tensor(1.4948), tensor(1.4706), tensor(1.4719), tensor(1.4851), tensor(1.4819), tensor(1.4978), tensor(1.5186), tensor(1.5325), tensor(1.5627), tensor(1.5845), tensor(1.6121), tensor(1.6326), tensor(1.6506), tensor(1.6736), tensor(1.6844), tensor(1.7056)"

q1_val_loss = clean_loss(q1_val_loss)

plt.figure(figsize=(10, 5))
plt.plot(iteration_numbers, q1_val_loss, label='Validation Loss')
plt.xlabel('Iteration Number')
plt.ylabel('Loss')
plt.title('Q1 Validation Loss Over Iterations')
plt.legend()
plt.grid(True)
# plt.savefig('q1_val_loss.png')


# Question 2
q2_32_val_loss = "tensor(4.1542), tensor(2.0977), tensor(1.7253), tensor(1.5825), tensor(1.5228), tensor(1.4849), tensor(1.4743), tensor(1.4739), tensor(1.4644), tensor(1.4791), tensor(1.4971), tensor(1.5167), tensor(1.5314), tensor(1.5476), tensor(1.5718), tensor(1.5981), tensor(1.6243), tensor(1.6377), tensor(1.6596), tensor(1.6679), tensor(1.6843)"
q2_8_val_loss = "tensor(4.2489), tensor(2.1539), tensor(1.7526), tensor(1.6001), tensor(1.5242), tensor(1.4970), tensor(1.4804), tensor(1.4695), tensor(1.4699), tensor(1.4777), tensor(1.4932), tensor(1.4982), tensor(1.5207), tensor(1.5388), tensor(1.5595), tensor(1.5726), tensor(1.5998), tensor(1.6090), tensor(1.6354), tensor(1.6452), tensor(1.6581)"

q2_32_val_loss = clean_loss(q2_32_val_loss)
q2_8_val_loss = clean_loss(q2_8_val_loss)

plt.figure(figsize=(10, 5))
plt.plot(iteration_numbers, q1_val_loss, label='Size 64 Validation Loss')
plt.plot(iteration_numbers, q2_32_val_loss, label='Size 32 Validation Loss')
plt.plot(iteration_numbers, q2_8_val_loss, label='Size 8 Validation Loss')
plt.xlabel('Iteration Number')
plt.ylabel('Loss')
plt.title('Q2 Validation Loss Over Iterations')
plt.legend()
plt.grid(True)
plt.savefig('q2_val_loss.png')
# plt.show()


# Question 3
q3_100_val_loss = "tensor(4.2843), tensor(2.0445), tensor(1.7174), tensor(1.5908), tensor(1.5226), tensor(1.5022), tensor(1.4810), tensor(1.4733), tensor(1.4773), tensor(1.4849), tensor(1.5027), tensor(1.5110), tensor(1.5304), tensor(1.5589), tensor(1.5804), tensor(1.6138), tensor(1.6352), tensor(1.6573), tensor(1.6826), tensor(1.6909), tensor(1.7119)"
q3_10_val_loss = "tensor(4.2727), tensor(1.8281), tensor(1.6228), tensor(1.5498), tensor(1.5127), tensor(1.5012), tensor(1.4992), tensor(1.5031), tensor(1.5326), tensor(1.5518), tensor(1.5732), tensor(1.6147), tensor(1.6286), tensor(1.6861), tensor(1.7165), tensor(1.7425), tensor(1.7782), tensor(1.7971), tensor(1.8305), tensor(1.8438), tensor(1.8740)"
q3_3_val_loss = "tensor(4.2603), tensor(1.7308), tensor(1.5838), tensor(1.5386), tensor(1.5168), tensor(1.5061), tensor(1.5004), tensor(1.5134), tensor(1.5133), tensor(1.5335), tensor(1.5443), tensor(1.5626), tensor(1.5668), tensor(1.6030), tensor(1.6077), tensor(1.6337), tensor(1.6554), tensor(1.6715), tensor(1.6881), tensor(1.6937), tensor(1.7130)"

q3_100_val_loss = clean_loss(q3_100_val_loss)
q3_10_val_loss = clean_loss(q3_10_val_loss)
q3_3_val_loss = clean_loss(q3_3_val_loss)

plt.figure(figsize=(10, 5))
plt.plot(iteration_numbers, q3_100_val_loss, label='Window 100 Validation Loss')
plt.plot(iteration_numbers, q3_10_val_loss, label='Window 10 Validation Loss')
plt.plot(iteration_numbers, q3_3_val_loss, label='Window 3 Validation Loss')
plt.xlabel('Iteration Number')
plt.ylabel('Loss')
plt.title('Q3 Validation Loss Over Iterations')
plt.legend()
plt.grid(True)
plt.savefig('q3_val_loss.png')
# plt.show()


# Question 4
q4_threeLayers_val_loss = "tensor(4.2561), tensor(1.9460), tensor(1.6356), tensor(1.5592), tensor(1.5338), tensor(1.4949), tensor(1.4893), tensor(1.4936), tensor(1.5177), tensor(1.5532), tensor(1.5964), tensor(1.6425), tensor(1.6976), tensor(1.7445), tensor(1.8038), tensor(1.8471), tensor(1.8938), tensor(1.9353), tensor(1.9708), tensor(2.0051), tensor(2.0190)"

q4_threeLayers_val_loss = clean_loss(q4_threeLayers_val_loss)

plt.figure(figsize=(10, 5))
plt.plot(iteration_numbers, q1_val_loss, label='Original Validation Loss')
plt.plot(iteration_numbers, q4_threeLayers_val_loss, label='Three Layers Validation Loss')
plt.xlabel('Iteration Number')
plt.ylabel('Loss')
plt.title('Q4 Validation Loss Over Iterations')
plt.legend()
plt.grid(True)
plt.savefig('q4_val_loss.png')
# plt.show()


# Question 5
q5_n_regist_5_val_loss = "tensor(4.2673), tensor(2.0648), tensor(1.7209), tensor(1.5773), tensor(1.5077), tensor(1.4900), tensor(1.4743), tensor(1.4746), tensor(1.4789), tensor(1.4674), tensor(1.4897), tensor(1.5059), tensor(1.5225), tensor(1.5609), tensor(1.5735), tensor(1.6064), tensor(1.6224), tensor(1.6347), tensor(1.6637), tensor(1.6814), tensor(1.6929)"
q5_n_regist_1_val_loss = "tensor(4.2285), tensor(2.0685), tensor(1.7328), tensor(1.5839), tensor(1.5240), tensor(1.4951), tensor(1.4840), tensor(1.4806), tensor(1.4828), tensor(1.4935), tensor(1.5040), tensor(1.5137), tensor(1.5404), tensor(1.5617), tensor(1.5771), tensor(1.5987), tensor(1.6269), tensor(1.6447), tensor(1.6723), tensor(1.6833), tensor(1.6936)"

q5_n_regist_5_val_loss = clean_loss(q5_n_regist_5_val_loss)
q5_n_regist_1_val_loss = clean_loss(q5_n_regist_1_val_loss)

plt.figure(figsize=(10, 5))
plt.plot(iteration_numbers, q1_val_loss, label='Original Validation Loss')
plt.plot(iteration_numbers, q5_n_regist_5_val_loss, label='n_regist 5 Validation Loss')
plt.plot(iteration_numbers, q5_n_regist_1_val_loss, label='n_regist 1 Validation Loss')
plt.xlabel('Iteration Number')
plt.ylabel('Loss')
plt.title('Q5 Validation Loss Over Iterations')
plt.legend()
plt.grid(True)
plt.savefig('q5_val_loss.png')
# plt.show()

# Question 6
q6_val_loss = "tensor(4.2230), tensor(1.7281), tensor(1.5866), tensor(1.5383), tensor(1.5054), tensor(1.5096), tensor(1.5072), tensor(1.5179), tensor(1.5172), tensor(1.5266), tensor(1.5429), tensor(1.5631), tensor(1.5694), tensor(1.5948), tensor(1.6089), tensor(1.6310), tensor(1.6533), tensor(1.6633), tensor(1.6978), tensor(1.6942), tensor(1.7128)"

q6_val_loss = clean_loss(q6_val_loss)

plt.figure(figsize=(10, 5))
plt.plot(iteration_numbers, q3_3_val_loss, label='Window 3 Validation Loss')
plt.plot(iteration_numbers, q5_n_regist_1_val_loss, label='n_regist 1 Validation Loss')
plt.plot(iteration_numbers, q6_val_loss, label='Validation Loss')
plt.xlabel('Iteration Number')
plt.ylabel('Loss')
plt.title('Q6 Validation Loss Over Iterations')
plt.legend()
plt.grid(True)
plt.savefig('q6_val_loss.png')
# plt.show()

# Question 7
q7_abs_val_loss = "tensor(4.2823), tensor(2.0662), tensor(1.7465), tensor(1.5872), tensor(1.5183), tensor(1.5000), tensor(1.4812), tensor(1.4642), tensor(1.4828), tensor(1.4814), tensor(1.5012), tensor(1.5162), tensor(1.5282), tensor(1.5550), tensor(1.5793), tensor(1.6055), tensor(1.6272), tensor(1.6485), tensor(1.6708), tensor(1.6798), tensor(1.7013)"

q7_abs_val_loss = clean_loss(q7_abs_val_loss)

plt.figure(figsize=(10, 5))
plt.plot(iteration_numbers, q1_val_loss, label='Exp Validation Loss')
plt.plot(iteration_numbers, q7_abs_val_loss, label='Absolute Validation Loss')
plt.xlabel('Iteration Number')
plt.ylabel('Loss')
plt.title('Q7 Validation Loss Over Iterations')
plt.legend()
plt.grid(True)
plt.savefig('q7_val_loss.png')
# plt.show()
