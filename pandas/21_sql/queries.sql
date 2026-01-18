create database pandas_sql;
use pandas_sql;

CREATE TABLE customers (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    phone_number VARCHAR(25) -- Best practice for phone numbers
);

insert into customers(id,name,phone_number)
values(1,"Donald","7326784567"),
(2,"Bill",6573489999),
(3,"Modi",4567895646);

select * from customers;

CREATE TABLE Orders(
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    customer_id INT NOT NULL,
    amount DECIMAL(10, 2) NOT NULL,
    -- You can also add a CHECK constraint to ensure non-negative amounts
    CHECK (amount >= 0), 
	FOREIGN KEY (customer_id) REFERENCES Customers(id)
	
);

insert into orders(id,name,amount,customer_id)
values(1,"Yoga Mat",20,2),
(2,"Google Pixel",950,1),
(3,"Fossil Watch",120,3);

select * from orders;

select customers.name,customers.phone_number,orders.name,orders.amount
from customers inner join orders
on customers.id=orders.customer_id