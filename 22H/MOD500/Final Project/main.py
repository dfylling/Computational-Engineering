#%%
import numpy as np
import matplotlib.pyplot as plt
#%%
np.random.seed(13)
N = 1000
# Uncertainties - triangular shapes [min, mode, max]
sales_price = np.array([1600,2000,2400])        # sales price per item
sales_volume = np.array([10,80,200])            # items sold per month
marketing_cost = np.array([20000,40000,50000])  # NOK per month
base_workload = np.array([100,150,300])         # hours per month
work_per_item = np.array([0.1,0.5,1])         # hours per sold item

base_production_cost = 500 # in range [0,200] products produced per month there will be a discount of 0.5 NOK per produced item
base_shipping_cost = 450 # if more than 100 products are shipped per month there will be a 100 NOK discount per item
owners_of_company = 2
consultant_hours = 0 # depending on total workload some work may be externalized
consultant_rate = 400 # NOK per hour

# Combinations of successful workload and salaries - [hours per month],[NOK per month]
# based on questionaire presented to partners
success_definition = np.array([[0,40,80,160],[20000,25000,35000,80000]])

record = np.empty([7, N])
key = ['Sales price', 'Sales volume', 'Marketing cost', 'Base workload', 'Work per sold item', 'Working hours per partner per month', 'Salary per partner per month']
mask = np.full(N, False)

def triangular_draw(f):
    return np.random.triangular(f[0],f[1],f[2])

# Monte Carlo style simulation collects random draws of variables, generates "salary" and "workload", and records all parameters for later use.
for i in range(N):
    s_p = triangular_draw(sales_price)
    s_v = triangular_draw(sales_volume)
    m_c = triangular_draw(marketing_cost)
    b_w = triangular_draw(base_workload)
    w_p_i = triangular_draw(work_per_item)

    p_c = base_production_cost - s_v*0.5

    if s_v > 100:
        s_c = base_shipping_cost - 100
    else:
        s_c = base_shipping_cost - 100
    
    total_workload = b_w + w_p_i*s_v

    if total_workload > 320:
        consultant_hours = total_workload - 320
        total_workload = 320
    
    workload = total_workload / owners_of_company
    
    income = (s_p-p_c-s_c)*s_v-m_c - consultant_hours*consultant_rate

    salary = income / owners_of_company

    record[0,i] = s_p
    record[1,i] = s_v
    record[2,i] = m_c
    record[3,i] = b_w
    record[4,i] = w_p_i
    record[5,i] = workload
    record[6,i] = salary

# Generate Boolean mask to split dataset in successful outcomes and unsuccessfull ones.
for i in range(N):
    if record[6,i] > np.interp(record[5,i], success_definition[0,:], success_definition[1,:]):
        mask[i] = True

plt.rcParams['figure.figsize'] = [12, 7]
plt.plot(record[5,mask], record[6,mask],'r*', label='Successful outcomes')
plt.plot(success_definition[0,:],success_definition[1,:],'r-', label='Successful outcomes - Limit')
plt.plot(record[5,np.logical_not(mask)], record[6,np.logical_not(mask)],'b.', label='Rejectable outcomes')
plt.plot(np.mean(record[5,:]), np.mean(record[6,:]),'go', markersize = 12, label = 'Expected value')
plt.xlabel('Working hours per month per partner')
plt.ylabel('Salary per month per partner')
plt.grid()
plt.legend() 
plt.show()
print("""We observe that the mean / expected result lies below the line defining a successful outcome for the owners.
        After facing the partners with the results, they seem thrilled, and eager to carry on.
        What does this mean? Find a way to define the utility function / risk attitude of the partners.
        """)

#%%

print(f'Chance of achieving goal is {np.count_nonzero(mask)/N*100:.1f} %')
print(f'Given goal is achieved:')
print(f'Mean expected workload is {np.mean(record[5,mask]):.0f} hours per partner per month')
print(f'With standard deviation {np.std(record[5,mask]):.0f}')
print(f'Mean expected salary is {np.mean(record[6,mask]):.0f} NOK per partner per month')
print(f'With standard deviation {np.std(record[6,mask]):.0f}')
print(f'---------------------')
print(f'Chance of not achieving goal is {(N-np.count_nonzero(mask))/N*100:.1f} %')
print(f'Given goal is not achieved:')
print(f'Mean expected workload is {np.mean(record[5,np.logical_not(mask)]):.0f} hours per partner per month')
print(f'With standard deviation {np.std(record[5,np.logical_not(mask)]):.0f}')
print(f'Mean expected salary is {np.mean(record[6,np.logical_not(mask)]):.0f} NOK per partner per month')
print(f'With standard deviation {np.std(record[6,np.logical_not(mask)]):.0f}')

#%%

print("""The alternative way for the partners to earn a living is to work as engineers at 350 NOK / hour.
        Subtracting this base case form the expected salary from the business venture gives
        the relative gain from base case. 
        """)
p = np.count_nonzero(mask)/N
w_Y = np.mean(record[5,mask])
s_Y = np.mean(record[6,mask])
w_N = np.mean(record[5,np.logical_not(mask)])
s_N = np.mean(record[6,np.logical_not(mask)])

hourly_wage = 350

gain_Y = s_Y-(w_Y*hourly_wage)
gain_N = s_N-(w_N*hourly_wage)

gain_Y_50 = gain_Y*p/0.5
gain_N_50 = gain_N*(1-p)/0.5
print(f"""After subtracting nominal wage the result is: +{gain_Y:.0f}/{gain_N:.0f} NOK.
        """)

#%%

print("Visualizing data distribution for each variable")
rows = 8
cols = 2
colors = ['b','r']
fig=plt.figure(figsize=(10,15))
masks = [np.full(N, True), mask]
labels = ['Full dataset', 'Acceptable outcomes']

for i in range(len(record[:-2,0])):
    ax=fig.add_subplot(rows,cols,i+1)
    ax.set_title(key[i])
    ax.set_axisbelow(True)
    ax.grid(color='whitesmoke')
    for m , mask in enumerate(masks):
        plt.hist(record[i,mask], color=colors[m], edgecolor='k', label = labels[m])
    plt.legend()
plt.tight_layout()  
plt.show()

# %%

print("Visualizing data distribution as errorbars")
rows = 8
cols = 2
colors = ['b','r']
fig=plt.figure(figsize=(10,15))
masks = [np.full(N, True), mask]
labels = ['Full dataset', 'Acceptable outcomes']

for i in range(len(record[:-2,0])):
    ax=fig.add_subplot(rows,cols,i+1)
    ax.set_title(key[i])
    ax.set_axisbelow(True)
    ax.grid(color='whitesmoke')
    
    plt.xlim(-1, 5)
    plt.xticks(ticks=[],labels=[])
    for m , mask in enumerate(masks):
        plt.errorbar(m, np.mean(record[i,mask]), np.std(record[i,mask]), color=colors[m],linestyle='None', marker='^', capsize=3, label = labels[m])     #.hist(bins=20,ax=ax,, alpha=0.6, )
    plt.legend()
plt.tight_layout()  
plt.show()

# %%