--DbDesign

DROP TABLE Paper_citation_num_history;
DROP TABLE Journal_impact_factor_history;
DROP TABLE	Account_paywallpaper_link;
DROP TABLE Account;
DROP TABLE Paywall_paper;
DROP TABLE Opensource_paper;
DROP TABLE LCN_collaborator;
DROP TABLE LCN_member;
DROP TABLE LCN_faculty;
DROP TABLE Researcher_paper_link;
DROP TABLE Researcher;
DROP TABLE Paper_topic_paper_link;
DROP TABLE Paper_topic;
DROP TABLE Paper;
DROP TABLE Journal;
DROP TABLE Journal_field;


CREATE TABLE Journal_field (
	Journal_field_id	DECIMAL(12)		NOT NULL	PRIMARY KEY,
	Journal_field_name	CHAR(100)		NOT NULL	);

CREATE TABLE Journal (
	Journal_id			DECIMAL(12) 	NOT NULL 	PRIMARY KEY,
	Journal_field_id	DECIMAL(12)		NOT NULL,		
	Journal_name 		VARCHAR(200) 	NOT NULL,	
	Impact_factor 		DECIMAL(5,3),
	FOREIGN KEY (Journal_field_id) REFERENCES Journal_field(Journal_field_id));

CREATE TABLE Paper (
	Paper_id 			DECIMAL(12)		NOT NULL 	PRIMARY KEY,
	Journal_id			DECIMAL(12) 	NOT NULL,
	Paper_name			VARCHAR(500) 	NOT NULL,
	Paper_year			DECIMAL(4)		NOT NULL,
	Reference			VARCHAR(500)	NOT NULL,
	DOI					VARCHAR(200),
	Paper_link			VARCHAR(500) 	NOT NULL,
	Citation_num		DECIMAL(6)		NOT NULL,
	PMCID 				VARCHAR(20),
	PMID				VARCHAR(20),
	FOREIGN KEY (Journal_id) REFERENCES Journal(Journal_id));

CREATE TABLE Paper_topic (
	Paper_topic_id		DECIMAL(12) 	NOT NULL 	PRIMARY KEY,
	Paper_topic_name 	CHAR(100)		NOT NULL);

CREATE TABLE Paper_topic_paper_link (
	Paper_topic_id		DECIMAL(12) 	NOT NULL,
	Paper_id			DECIMAL(12) 	NOT NULL,
	FOREIGN KEY (Paper_topic_id) REFERENCES Paper_topic(Paper_topic_id),
	FOREIGN KEY (Paper_id) REFERENCES Paper(Paper_id));

CREATE TABLE Researcher (
	Researcher_id		DECIMAL(12)		NOT NULL	PRIMARY KEY,
	First_name			CHAR(200)		NOT NULL,
	Mid_name			CHAR(200),
	Last_name			CHAR(200));

CREATE TABLE Researcher_paper_link (
	Researcher_id		DECIMAL(12) 	NOT NULL,
	Paper_id			DECIMAL(12) 	NOT NULL,
	FOREIGN KEY (Researcher_id) REFERENCES Researcher(Researcher_id),
	FOREIGN KEY (Paper_id) REFERENCES Paper(Paper_id));

CREATE TABLE LCN_faculty (
	Researcher_id		DECIMAL(12)		NOT NULL	PRIMARY KEY,
	FOREIGN KEY (Researcher_id) REFERENCES Researcher(Researcher_id));

CREATE TABLE LCN_member (
	Researcher_id		DECIMAL(12)		NOT NULL	PRIMARY KEY,
	FOREIGN KEY (Researcher_id) REFERENCES Researcher(Researcher_id));

CREATE TABLE LCN_collaborator (
	Researcher_id		DECIMAL(12)		NOT NULL	PRIMARY KEY,
	FOREIGN KEY (Researcher_id) REFERENCES Researcher(Researcher_id));

CREATE TABLE Opensource_paper (
	Paper_id			DECIMAL(12)		NOT NULL	PRIMARY KEY,
	Opensource_pdf		BYTEA,
	FOREIGN KEY (Paper_id) REFERENCES Paper(Paper_id));

CREATE TABLE Paywall_paper (
	Paper_id			DECIMAL(12)		NOT NULL	PRIMARY KEY,
	Paywall_paper_id	DECIMAL(12)		NOT NULL	UNIQUE,
	Paywall_pdf			BYTEA,
	FOREIGN KEY (Paper_id) REFERENCES Paper(Paper_id));

CREATE TABLE Account (
	Account_id			DECIMAL(12)		PRIMARY KEY,
	Username			VARCHAR(64),
	Email				VARCHAR(64)		NOT NULL,
	Password			VARCHAR(64)		NOT NULL);

CREATE TABLE Account_paywallpaper_link (
	Account_id			DECIMAL(12) 	NOT NULL,
	Paywall_paper_id	DECIMAL(12) 	NOT NULL,
	FOREIGN KEY (Account_id) REFERENCES Account(Account_id),
	FOREIGN KEY (Paywall_paper_id) REFERENCES Paywall_paper(Paywall_paper_id));


--Foreign Key Non-Unique INDEXES
CREATE INDEX JournalFieldIdx
ON Journal(Journal_field_id);

CREATE INDEX ResPapLinkPaperIdx
ON Researcher_paper_link(Paper_id);

CREATE INDEX ResPapLinkResearcherIdx
ON Researcher_paper_link(Researcher_id);

CREATE INDEX PapTopPapLinkPapTopIdx
ON Paper_topic_paper_link(Paper_topic_id);

CREATE INDEX PapTopPapLinkPapIdx
ON Paper_topic_paper_link(Paper_id);

CREATE INDEX PaperJournalIdx
ON Paper(Journal_id);

CREATE INDEX AccPaywallPapLinkAccIdx
ON Account_paywallpaper_link(Account_id);

CREATE INDEX AccPaywallPapLinkPaywallPapIdx
ON Account_paywallpaper_link(Paywall_paper_id);


--Foreign Key Unique INDEXES
CREATE UNIQUE INDEX LcnFacultyResearcherIdx
ON LCN_faculty(Researcher_id);

CREATE UNIQUE INDEX LcnMemberResearcherIdx
ON LCN_member(Researcher_id);

CREATE UNIQUE INDEX LcnCollabResearcherIdx
ON LCN_collaborator(Researcher_id);

CREATE UNIQUE INDEX PaywallPapPaperIdx
ON Paywall_paper(Paper_id);

CREATE UNIQUE INDEX OpensourcePapPaperIdx
ON Opensource_paper(Paper_id);


--Query driven INDEXES

CREATE INDEX PaperPapYearIdx
ON Paper(Paper_year);

CREATE INDEX PaperCitationNumIdx
ON Paper(Citation_num);

CREATE INDEX JournalImpactIdx
ON Journal(Impact_factor);


--Journal impact factor history Trigger
--function
CREATE TABLE Journal_impact_factor_history (
	Impact_history_id DECIMAL(12) NOT NULL PRIMARY KEY,
	Journal_id DECIMAL(12) NOT NULL,
	Old_impact_factor DECIMAL(5,3) NOT NULL,
	New_impact_factor DECIMAL(5,3) NOT NULL,
	Change_date DATE NOT NULL,
	FOREIGN KEY (Journal_id) REFERENCES Journal(Journal_id));


--trigger
CREATE OR REPLACE FUNCTION Journal_impact_factor_history_func() RETURNS TRIGGER LANGUAGE plpgsql
AS $trigfunc$
  BEGIN
  	IF OLD.Impact_factor != NEW.Impact_factor THEN
		INSERT INTO Journal_impact_factor_history(Impact_history_id, Journal_id, 
					 Old_impact_factor, New_impact_factor, Change_date)
		VALUES(COALESCE((SELECT MAX(Impact_history_id)+1 FROM Journal_impact_factor_history), 50001), NEW.Journal_id, 
			   OLD.Impact_factor, NEW.Impact_factor, current_date);
  		END IF;
  RETURN NEW;
  END;
$trigfunc$;
CREATE TRIGGER Journal_impact_factor_history_trig
BEFORE UPDATE OF Impact_factor ON Journal
FOR EACH ROW
EXECUTE PROCEDURE Journal_impact_factor_history_func();


/*
--Journal_impact_factor_history_trig trigger testing
INSERT INTO Journal_field VALUES (10501, 'Biochemistry, Genetics and Molecular Biology');
INSERT INTO Journal_field VALUES (10504, 'Health Professions');
INSERT INTO Journal_field VALUES (10506, 'Medicine');

INSERT INTO Journal VALUES(11002, 10504, 'Advances in Alzheimer''s Disease', 2.13);
INSERT INTO Journal VALUES(11003, 10506, 'American Journal of Neuroradiology', 3.653);
INSERT INTO Journal VALUES(11004, 10506, 'Annals of Neurology', 10.244);
INSERT INTO Journal VALUES(11005, 10501, 'Annals of the New York Academy of Sciences', 4.039);


UPDATE Journal
SET Impact_factor = 5.44
WHERE Journal_id = 11002;

UPDATE Journal
SET Impact_factor = 4.345
WHERE Journal_id = 11002;

UPDATE Journal
SET Impact_factor = 5.785
WHERE Journal_id = 11003;

UPDATE Journal
SET Impact_factor = 8.438
WHERE Journal_id = 11003;

UPDATE Journal
SET Impact_factor = 1.765
WHERE Journal_id = 11004;

UPDATE Journal
SET Impact_factor = 2.098
WHERE Journal_id = 11005;


SELECT *
FROM Journal_impact_factor_history;
*/


--sometimes triggers can only run one at a time if there are too many transactions processing 
--comment out triggers that come after a Select statement to get data output


--Paper citation number history Trigger
--function
CREATE TABLE Paper_citation_num_history (
	Citation_history_id DECIMAL(12) NOT NULL PRIMARY KEY,
	Paper_id DECIMAL(12) NOT NULL,
	Old_citation_num DECIMAL(6) NOT NULL,
	New_citation_num DECIMAL(6) NOT NULL,
	Change_date DATE NOT NULL,
	FOREIGN KEY (Paper_id) REFERENCES Paper(Paper_id));


--trigger
CREATE OR REPLACE FUNCTION Paper_citation_num_history_func() RETURNS TRIGGER LANGUAGE plpgsql
AS $trigfunc$
  BEGIN
  --add if to reject if New.citation is less than old.citation
  	IF OLD.Citation_num != NEW.Citation_num THEN
		INSERT INTO Paper_citation_num_history(Citation_history_id, Paper_id, 
					 Old_citation_num, New_citation_num, Change_date)
   		VALUES(COALESCE((SELECT MAX(Citation_history_id)+1 FROM Paper_citation_num_history), 40001), NEW.Paper_id, 
			   OLD.Citation_num, NEW.Citation_num, current_date);
		END IF;
  RETURN NEW;
  END;
$trigfunc$;
CREATE TRIGGER Paper_citation_num_history_trig
BEFORE UPDATE OF Citation_num ON Paper
FOR EACH ROW
EXECUTE PROCEDURE Paper_citation_num_history_func();


/*
--Paper_citation_num_history_trig trigger testing
INSERT INTO Journal_field VALUES (10507, 'Neuroscience');
INSERT INTO Journal VALUES(11001, 10507, 'Acta Neuropathologica', 18.17);
INSERT INTO Paper VALUES(20001, 11001, 'Shared genetic risk between corticobasal degeneration, progressive supranuclear palsy, and 
						 					frontotemporal dementia', 2017, 'Yokoyama, J.S., Karch, C.M., Fan, C.C. and 28 more (...) 
						 					(2017).Shared genetic risk between corticobasal degeneration, progressive supranuclear palsy, 
						 					and frontotemporal dementia. Acta Neuropathologica,133(5) 825-837', '10.1007/s00401-017-1693-y',
						 					'https://www.scopus.com/record/display.url?eid=2-s2.0-85014531490&origin=resultslist', 31, '', '');

--test trigger - paper citation number
UPDATE Paper
SET Citation_num = 46
WHERE Paper_id = 20001;

SELECT *
FROM Paper_citation_num_history;
*/


--COMMIT;

/*
--use select statements to test that tables have been created
SELECT *
FROM LCN_collaborator;
*/

COMMIT;