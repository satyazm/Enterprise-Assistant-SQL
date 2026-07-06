-- database/sample_db.sql
-- Schema matches the actual CSVs: customers.csv, products.csv,
-- sample_sales.csv, complaints.csv. Run this as the admin user
-- (raguser) before loading data.

DROP TABLE IF EXISTS complaints CASCADE;
DROP TABLE IF EXISTS sales CASCADE;
DROP TABLE IF EXISTS products CASCADE;
DROP TABLE IF EXISTS customers CASCADE;

CREATE TABLE customers (
    customer_id     VARCHAR(20) PRIMARY KEY,
    first_name      VARCHAR(100) NOT NULL,
    last_name       VARCHAR(100) NOT NULL,
    email           VARCHAR(255) NOT NULL,
    phone           VARCHAR(30),
    street_address  VARCHAR(255),
    city            VARCHAR(100),
    state           VARCHAR(10),
    zip_code        INTEGER,
    region          VARCHAR(50),
    loyalty_tier    VARCHAR(20),
    signup_date     DATE
);

CREATE TABLE products (
    product_id      VARCHAR(20) PRIMARY KEY,
    product_name    VARCHAR(255) NOT NULL,
    product_line    VARCHAR(100),
    category        VARCHAR(100),
    unit_price      NUMERIC(10, 2),
    unit_cost       NUMERIC(10, 2),
    margin_pct      NUMERIC(5, 2)
);

CREATE TABLE sales (
    sale_id         VARCHAR(20) PRIMARY KEY,
    sale_date       DATE NOT NULL,
    customer_id     VARCHAR(20) NOT NULL REFERENCES customers(customer_id),
    product_id      VARCHAR(20) NOT NULL REFERENCES products(product_id),
    product_name    VARCHAR(255),
    product_line    VARCHAR(100),
    quantity        INTEGER NOT NULL,
    unit_price      NUMERIC(10, 2),
    discount_pct    INTEGER,
    line_total      NUMERIC(10, 2),
    channel         VARCHAR(50),
    region          VARCHAR(50)
);

CREATE TABLE complaints (
    complaint_id        VARCHAR(20) PRIMARY KEY,
    complaint_date      DATE NOT NULL,
    customer_id         VARCHAR(20) NOT NULL REFERENCES customers(customer_id),
    product_id          VARCHAR(20) NOT NULL REFERENCES products(product_id),
    product_name        VARCHAR(255),
    product_line        VARCHAR(100),
    category            VARCHAR(150),
    channel             VARCHAR(50),
    status              VARCHAR(50),
    resolution_days     NUMERIC(5, 1),   -- NULL = not yet resolved
    satisfaction_score  NUMERIC(3, 1)    -- NULL = not yet resolved
);

-- Indexes on commonly-filtered columns so generated SQL doesn't full-scan
CREATE INDEX idx_sales_region ON sales(region);
CREATE INDEX idx_sales_date ON sales(sale_date);
CREATE INDEX idx_sales_customer ON sales(customer_id);
CREATE INDEX idx_sales_product ON sales(product_id);

CREATE INDEX idx_complaints_date ON complaints(complaint_date);
CREATE INDEX idx_complaints_customer ON complaints(customer_id);
CREATE INDEX idx_complaints_product ON complaints(product_id);
CREATE INDEX idx_complaints_status ON complaints(status);

CREATE INDEX idx_customers_region ON customers(region);
