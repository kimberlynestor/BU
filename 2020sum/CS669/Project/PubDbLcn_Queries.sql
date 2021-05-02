--DbQueries

--QUERIES
--only the last query will print, comment out other queries

--db commit test
SELECT*
FROM Paper;


--all papers and all the authors
SELECT paper_name, last_name, first_name
FROM Researcher_paper_link
LEFT JOIN Paper ON Researcher_paper_link.Paper_id = Paper.Paper_id
LEFT JOIN Researcher ON Researcher_paper_link.Researcher_id = Researcher.Researcher_id;


--papers and their associated topics
SELECT Paper_name, Paper_topic_name
FROM Paper
RIGHT JOIN Paper_topic_paper_link ON Paper_topic_paper_link.Paper_id = Paper.Paper_id
JOIN Paper_topic ON Paper_topic_paper_link.Paper_topic_id = Paper_topic.Paper_topic_id;


--LCN_faculty only
SELECT First_name, Mid_name, Last_name 
FROM Researcher
RIGHT JOIN LCN_faculty ON Researcher.Researcher_id = LCN_faculty.Researcher_id;


--Which researchers are LCN_faculty? What papers have they publsihed in the last ten years? Paper topics listed for reference.
SELECT Researcher.First_name, Researcher.Mid_name, Researcher.Last_name, Paper.Paper_name, Paper.Paper_year, Paper_topic
FROM Researcher
RIGHT JOIN LCN_faculty ON Researcher.Researcher_id = LCN_faculty.Researcher_id
JOIN Researcher_paper_link ON Researcher_paper_link.Researcher_id = Researcher.Researcher_id
JOIN Paper ON Researcher_paper_link.Paper_id = Paper.Paper_id
LEFT JOIN Paper_topic_paper_link ON Paper_topic_paper_link.Paper_id = Paper.Paper_id
LEFT JOIN Paper_topic ON Paper_topic_paper_link.Paper_topic_id = Paper_topic.Paper_topic_id
WHERE Paper_year >= 2010
ORDER BY Last_name;


--What are the top five journals with the highest impact? Which researchers are in these journals and what papers did they publish? 
--Impact factor and journal topic included for reference.
SELECT Researcher.Last_name, Researcher.First_name, Paper.Paper_name, Journal.Journal_name, Journal_field_name, Journal.Impact_factor
FROM Researcher
JOIN Researcher_paper_link ON Researcher_paper_link.Researcher_id = Researcher.Researcher_id
JOIN Paper ON Researcher_paper_link.Paper_id = Paper.Paper_id
RIGHT JOIN Journal on Journal.Journal_id = Paper.Journal_id
JOIN Journal_field ON Journal_field.Journal_field_id = Journal.Journal_field_id
WHERE Journal.Journal_id IN (SELECT Journal.Journal_id
						FROM Journal
						JOIN Journal_field ON Journal_field.Journal_field_id = Journal.Journal_field_id
						WHERE Journal.Impact_factor IS NOT NULL
						ORDER BY Journal.Impact_factor DESC
						LIMIT 5)
GROUP BY researcher.last_name, Researcher.First_name, Paper.Paper_name, Journal.Journal_name, Journal_field_name, 
		  Journal.Impact_factor, Journal.Journal_id
ORDER BY journal.impact_factor DESC;


--What journals have lost impact? Listing only journals with new impact less than 2.
SELECT Journal.Journal_name, old_impact_factor, new_impact_factor, change_date
FROM Journal_impact_factor_history
JOIN Journal ON Journal.Journal_id = Journal_impact_factor_history.Journal_id
WHERE new_impact_factor < old_impact_factor
GROUP BY Journal.journal_name, old_impact_factor, new_impact_factor, change_date
HAVING new_impact_factor < 2;


/*
--uncorrelated subquery for above full query
SELECT Journal.Journal_id, Journal_name, Journal.Impact_factor
FROM Journal
JOIN Journal_field ON Journal_field.Journal_field_id = Journal.Journal_field_id
WHERE Journal.Impact_factor IS NOT NULL
ORDER BY Journal.Impact_factor DESC
LIMIT 10;
*/

