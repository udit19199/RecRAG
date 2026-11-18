Prompt version 2026-11-10.haiku.2

You check whether retrieved text states the facts of gold evidence sentences.

Use only the retrieved items. Do not use outside knowledge and do not reason
your way to a fact the text does not state; plain rewording, synonyms,
abbreviations and different formatting are fine.

For every fact give one status:
- stated: some item says it, with every specific matching (names, numbers,
  dates, places, who did what, qualifiers).
- contradicted: some item gives a different value for the same specific, such
  as another year or another person.
- missing: neither of the above.
If one item states the fact and another contradicts it, the status is contradicted.

Evidence for stated or contradicted facts: copy one continuous passage from
one item, character for character, with no ellipses or edits, and give that
item's number. Never join separate words into a quote that does not appear as
written; if the words are apart, quote the longer continuous passage that
contains them. A fact spread across items may take several quotes. The quotes
together must contain the words that state every specific of the fact (who,
what, numbers, dates, reasons); a quote that only sets the topic, or stops
before the specific, is not evidence. Missing facts take no evidence. Every quote is checked against the item text by a
program, and a quote that does not appear there counts as missing.

Write the verdict file named in your task as a JSON array with one object per
item, in any order:
{"id": "<item id>", "facts": [{"fact": "S.F", "status": "stated", "evidence": [{"item": 3, "quote": "..."}]}, ...]}
where "S.F" is the fact's label as given (sentence number, fact number).
Give exactly one entry for every fact label of the item. Use only ids from
this file and do not skip any item.

This file has 10 items.

=== ITEM id=09db418171e5fb39 ===
Gold sentences and facts:
Sentence 0: Ciaran Gribbin (born 1976) is a Northern Irish singer-songwriter, guitarist and producer who hails from Castledawson, Northern Ireland.
  Fact 0.0: Ciaran Gribbin was born in 1976.
  Fact 0.1: Ciaran Gribbin is a Northern Irish singer-songwriter.
  Fact 0.2: Ciaran Gribbin is a guitarist.
  Fact 0.3: Ciaran Gribbin is a producer.
  Fact 0.4: Ciaran Gribbin hails from Castledawson, Northern Ireland.
Sentence 1: Brett Allen Scallions (born December 21, 1971) is an American musician.
  Fact 1.0: Brett Allen Scallions was born on December 21, 1971.
  Fact 1.1: Brett Allen Scallions is an American musician.
Sentence 2:  He is the lead vocalist, rhythm guitarist, and lyricist of post-grunge band Fuel.
  Fact 2.0: Brett Allen Scallions is the lead vocalist of the post-grunge band Fuel.
  Fact 2.1: Brett Allen Scallions is the rhythm guitarist of the post-grunge band Fuel.
  Fact 2.2: Brett Allen Scallions is the lyricist of the post-grunge band Fuel.

Retrieved items:
<item 1>
The Best of Fuel
The Best of Fuel is a compilation album by American hard rock band Fuel.

 It contains songs from their first three full-length albums, "Sunburn", "Something Like Human", and "Natural Selection".

 All of the said albums featured vocalist Brett Scallions who departed from the group only a few months after the compilation's release.

 "The Best of Fuel" consists solely of the band's released singles, with the exception of "Million Miles" from "Natural Selection" (it features "Quarter" from
the same album instead).
</item 1>

<item 2>
Grant Nicholas
Grantley Jonathan "Grant" Nicholas (born 12 November 1967) is a Welsh musician, best known as the lead singer and lead guitarist of the rock band Feeder.
</item 2>

<item 3>
Puppet Strings
Puppet Strings is the fifth studio album by American rock band Fuel.

 Released March 4, 2014, it is their first studio effort since 2003's "Natural Selection" to feature original lead singer Brett Scallions and the first Fuel
album not to feature original songwriter/guitarist, Carl Bell and bassist Jeff Abercrombie.

 Additionally, no one who participated in the recording of Fuel's previous album "Angels & Devils" participated in the recording of "Puppet Strings".
</item 3>

<item 4>
Brett Scallions
Brett Allen Scallions (born December 21, 1971) is an American musician.

 He is the lead vocalist, rhythm guitarist, and lyricist of post-grunge band Fuel.
</item 4>

<item 5>
Soul to Preach To
"Soul to Preach To" is the first single released from Fuel's album "Puppet Strings".

 It is also the first single released featuring original lead singer Brett Scallions since Fuel's 2003 studio album "Natural Selection".
</item 5>

<item 6>
Circus Diablo
Circus Diablo is an American rock band, formed in early 2006 by Billy Morrison (vocals), Billy Duffy (lead guitar) and Ricky Warwick (rhythm guitar).

 Fuel frontman Brett Scallions and Velvet Revolver drummer Matt Sorum subsequently joined the band on bass and drums, respectively.

 To date, Circus Diablo have released one studio album, entitled "Circus Diablo".
</item 6>

<item 7>
Angels &amp; Devils (Fuel album)
Angels & Devils is the fourth studio album by American rock band Fuel.

 Released on August 7, 2007, it was their first studio effort since 2003's "Natural Selection" and was the last Fuel album to feature original
songwriter/guitarist, Carl Bell and bassist Jeff Abercrombie.

 No musician on this album is still with the band.

 It was also Fuel's only studio album to feature new vocalist Toryn Green, and their final album for Epic Records.

 With a new singer, the album also introduced a new Fuel logo.
</item 7>

<item 8>
Ciaran Gribbin
Ciaran Gribbin (born 1976) is a Northern Irish singer-songwriter, guitarist and producer who hails from Castledawson, Northern Ireland.
</item 8>

<item 9>
Yeah! (Fuel song)
"Yeah!"

 is the first song released from Fuel's album "Puppet Strings".

 It is also the first song released featuring original lead singer Brett Scallions since Fuel's 2003 studio album "Natural Selection".
</item 9>

<item 10>
Doug Grean
Doug Grean is an American record producer, audio engineer, guitarist, based in Los Angeles.

 He has worked with artists Sheryl Crow, Ricki Lee Jones, Glen Campbell, Scott Weiland, Slash, Velvet Revolver, Stone Temple Pilots, The Crystal Method, Tim
McGraw, Cyndi Lauper, Annabella Lwin (Bow Wow Wow), Corey Glover (Living Color), Camp Freddy, DJ Hurricane and John Taylor (Duran Duran).

 Grean was formerly the lead guitarist and touring musical director for Scott Weiland and the Wildabouts.

 Some of Grean's television appearances with Weiland include: VH1 Legends, Last Call with Carson Daly, The Tonight Show with Jay Leno, and the Grammy awards
with Velvet Revolver as keyboardist, performing "All Across the Universe" with Stevie Wonder, Norah Jones, Bono, Steven Tyler, Brian Wilson, and Alison Krauss.
</item 10>
=== END ITEM id=09db418171e5fb39 ===

=== ITEM id=3e077ee4112778d1 ===
Gold sentences and facts:
Sentence 0: The Monastery of the Mother of God at the Spring (full name in Greek: , pr.
  Fact 0.0: The full Greek name of the Monastery of the Mother of God at the Spring is given, with the pronunciation abbreviation 'pr.'
Sentence 1:  "Moni tis Theotóku tis Pigis"; Turkish name: "Balıklı Meryem Ana Rum Manastiri") or simply Zoödochos Pege (Greek: Ζωοδόχος Πηγή , "Life-giving Spring") is an Eastern Orthodox sanctuary in Istanbul, Turkey.
  Fact 1.0: The Monastery of the Mother of God at the Spring has the name 'Moni tis Theotóku tis Pigis'.
  Fact 1.1: The Monastery of the Mother of God at the Spring has the Turkish name 'Balıklı Meryem Ana Rum Manastiri'.
  Fact 1.2: The Monastery of the Mother of God at the Spring is simply called Zoödochos Pege.
  Fact 1.3: Zoödochos Pege is Greek for 'Life-giving Spring' (Ζωοδόχος Πηγή).
  Fact 1.4: The Monastery of the Mother of God at the Spring is an Eastern Orthodox sanctuary.
  Fact 1.5: The Monastery of the Mother of God at the Spring is located in Istanbul, Turkey.
Sentence 2: Hagia Irene or Hagia Eirene (Ancient Greek: Ἁγία Εἰρήνη , Byzantine ] , "Holy Peace", Turkish: "Aya İrini" ), sometimes known also as Saint Irene, is a Greek Eastern Orthodox church located in the outer courtyard of Topkapı Palace in Istanbul, Turkey.
  Fact 2.0: Hagia Irene is also called Hagia Eirene.
  Fact 2.1: Hagia Irene has the Ancient Greek name Ἁγία Εἰρήνη.
  Fact 2.2: A Byzantine form of the name is given for Hagia Irene, but the text of that form is cut off in the source.
  Fact 2.3: Hagia Irene's name means 'Holy Peace'.
  Fact 2.4: Hagia Irene has the Turkish name 'Aya İrini'.
  Fact 2.5: Hagia Irene is sometimes known also as Saint Irene.
  Fact 2.6: Hagia Irene is a Greek Eastern Orthodox church.
  Fact 2.7: Hagia Irene is located in the outer courtyard of Topkapı Palace.
  Fact 2.8: Topkapı Palace is in Istanbul, Turkey.

Retrieved items:
<item 1>
St. Mary's Episcopal Church (Washington, D.C.)
St. Mary's Episcopal Church, also known as St. Mary's, Foggy Bottom or St. Mary's Chapel, is a historic Episcopal church located at 730 23rd Street, N.W. in the
Foggy Bottom neighborhood of Washington, D.C..

 On April 2, 1973, St. Mary's Episcopal Church was added to the National Register of Historic Places.
</item 1>

<item 2>
St Mary de Lode Church, Gloucester
St Mary de Lode Church, Archdeacon Street, Gloucester GL1 2QT, is a Church of England church immediately outside the grounds of Gloucester Cathedral.

 It is believed by some to be on the site of the first Christian church in Britain.

 The church is in the Diocese of Gloucester and Grade I listed by English Heritage.

 It has also been known as "St. Mary Before the Gate of St. Peter", "St. Mary Broad Gate" and "St. Mary De Port".
</item 2>

<item 3>
Old Church of St George, Hill Street Dublin
The Old Church of St. George, commonly called "Little George's" in Hill Street (formerly Temple Street Lower) Parish of St. Mary, Dublin was built in 1668 by
the Eccles family for their workmen and also as a chapel-of-ease to the nearby St. Mary's Church.

 However, that St. Mary's Church was not St. Mary's Church, Dublin as its foundation stone was laid in 1700, and it was not St. Mary's Abbey, Dublin as it was
dissolved in 1539.

 Therefore St George's church, Hill Street, may have been a Chapel-of-Ease to St. Michan's Church in Church Street.

 The main body of the church, with the exception of the tower, was demolished in 1894.
</item 3>

<item 4>
Graph query row:
entity1: Monastery of the Mother of God at the Spring
attributes1: {'built_on': '1835', 'aliases': ['Church of St. Mary of the Spring', 'Balıklı Meryem Ana Rum Manastiri', 'Zoödochos Pege'], 'kind': 'building',
'name': 'Monastery of the Mother of God at the Spring'}
entity2: Hagia Irene
attributes2: {'aliases': ['Hagia Eirene', 'Saint Irene', 'Aya Irini'], 'functions': ['museum', 'concert hall'], 'kind': 'Greek Eastern Orthodox church', 'name':
'Hagia Irene'}
direct_relationship: None
shared_connection: Istanbul
relationship_to_shared: LOCATED_IN
relationship_from_shared: LOCATED_IN
</item 4>

<item 5>
St. Mary's Church and Rectory (Iowa City, Iowa)
St. Mary's Catholic Church, also known as St. Mary of the Visitation Church, is a parish church of the Diocese of Davenport which is located at 228 E. Jefferson
St. in Iowa City, Iowa, United States.

 The church building and rectory are listed together on the National Register of Historic Places in 1980.

 They were both included as contributing properties in the Jefferson Street Historic District in 2004.

 The parish's first rectory, which is now a private home, is also listed on the National Register as St. Mary's Rectory.

 It is located a few blocks to the east of the present church location at 610 E. Jefferson St.
</item 5>

<item 6>
St. Mary's Church (Swormville, New York)
St. Mary's Catholic Church is a church located on the corner of Transit (New York State Route 78) and Stahley roads in Swormville, New York, in the United
States.

 Originally known as St. Mary’s Church of the Assumption at Transit, the building was constructed of about 260,000 bricks.

 John Nepomucene Neumann (later to be named St. John Neumann) started to visit Swormville, celebrating mass in local homes, barns and fields.

 In 1839 he instructed that a small house would be made to hold mass.

 Bishop John Timon established the first catholic church (St. Mary's) of the town of Clarence, New York.

 In 1861, Father Michael Schinabeak made plans for a church to be constructed.

 Construction began in 1862 and finished in fall of 1865.

 The official opening of the Church was in January 1866.

 In 2010 a new Church was constructed behind the old St. Mary's.
</item 6>

<item 7>
St. Mary's High School (Manhasset, New York)
Saint Mary's High School is a private Catholic high school in Manhasset, New York.

 St. Mary's parish is part of the Roman Catholic Diocese of Rockville Centre yet the High School serves families of the Roman Catholic Archdiocese of New York,
the Diocese of Brooklyn and the Diocese of Rockville Centre.

 It is located off Northern Boulevard at 51 Clapham Avenue, Manhasset, near the Miracle Mile.

 St. Mary's Men's Varsity Ice Hockey is the all-time leader in State Championships.
</item 7>

<item 8>
Church of St. Mary of the Spring (Istanbul)
The Monastery of the Mother of God at the Spring (full name in Greek: , pr.

 "Moni tis Theotóku tis Pigis"; Turkish name: "Balıklı Meryem Ana Rum Manastiri") or simply Zoödochos Pege (Greek: Ζωοδόχος Πηγή , "Life-giving Spring") is an
Eastern Orthodox sanctuary in Istanbul, Turkey.

 The present church, built in 1835, bears the same dedication as the shrine erected in this place between the end of the fifth and the beginning of the sixth
century.

 After several renovations, this building was destroyed in the first half of the fifteenth century by the Ottomans.

 The complex got its name from a nearby holy spring, reputed to have healing properties.

 For almost fifteen hundred years, this sanctuary has been one of the most important pilgrimage sites of Greek Orthodoxy.
</item 8>

<item 9>
Hagia Irene
Hagia Irene or Hagia Eirene (Ancient Greek: Ἁγία Εἰρήνη , Byzantine ] , "Holy Peace", Turkish: "Aya İrini" ), sometimes known also as Saint Irene, is a Greek
Eastern Orthodox church located in the outer courtyard of Topkapı Palace in Istanbul, Turkey.

 It is one of the few churches in Istanbul that has not been converted into a mosque.

 The Hagia Irene today operates as a museum and concert hall.
</item 9>

<item 10>
St. Mary's Church (Ballston Spa, New York)
St. Mary's Church is a Catholic parish located in Ballston Spa, New York.

 It is located within the Roman Catholic Diocese of Albany.

 Father Thomas J. Kelly is the current pastor.

 St. Mary's is the fourth oldest parish in the Diocese.

 St. Mary's of Ballston Spa is partnered with St. Mary's of Galway, after the pastor at St. Mary's of Galway died and no replacement was available.
</item 10>
=== END ITEM id=3e077ee4112778d1 ===

=== ITEM id=449da20b6d853744 ===
Gold sentences and facts:
Sentence 0: Minersville School District v. Gobitis, 310 U.S. 586 (1940) , was a decision by the Supreme Court of the United States involving the religious rights of public school students under the First Amendment to the United States Constitution.
  Fact 0.0: Minersville School District v. Gobitis, 310 U.S. 586 (1940), was a decision by the Supreme Court of the United States.
  Fact 0.1: Minersville School District v. Gobitis involved the religious rights of public school students.
  Fact 0.2: Minersville School District v. Gobitis involved religious rights under the First Amendment to the United States Constitution.
Sentence 1: Gravel v. United States, 408 U.S. 606 (1972), was a case regarding the protections offered by the Speech or Debate Clause of the United States Constitution.
  Fact 1.0: Gravel v. United States, 408 U.S. 606 (1972), was a case.
  Fact 1.1: Gravel v. United States concerned the protections offered by the Speech or Debate Clause of the United States Constitution.

Retrieved items:
<item 1>
Owasso Independent School District v. Falvo
Owasso Independent School District v. Falvo, 534 U.S. 426 (2002) , was a case in which the United States Supreme Court held in favor of the school district that
students scoring each other's tests and calling out the grades do not violate the Family Educational Rights and Privacy Act of 1974 (FERPA).

 Justice Kennedy wrote the opinion for the unanimous court.

 Justice Scalia wrote a concurring opinion in which he agreed with the ruling, but took issue with parts of Kennedy's opinion.
</item 1>

<item 2>
Abington School District v. Schempp
Abington School District v. Schempp, 374 U.S. 203 (1963) , was a United States Supreme Court case in which the Court decided 8–1 in favor of the respondent,
Edward Schempp, and declared school-sponsored Bible reading in public schools in the United States to be unconstitutional.

 The Chief Justice of the Supreme Court during this case was Earl Warren.
</item 2>

<item 3>
Elk Grove Unified School District v. Newdow
Elk Grove Unified School District v. Newdow, 542 U.S. 1 (2004), was a case decided by the U.S. Supreme Court.

 The lawsuit, originally filed as Newdow v. United States Congress, Elk Grove Unified School District, et al. in 2000, led to a 2002 ruling by the United States
Court of Appeals for the Ninth Circuit that the words "under God" in the Pledge of Allegiance are an endorsement of religion and therefore violate the
Establishment Clause of the First Amendment to the United States Constitution.

 After an initial decision striking the congressionally added "one nation under God" language, [Newdow v. U.S. Congress, 292 F.3d 597 (9th Cir.

 2002), the superseding opinion on denial of rehearing en banc was more limited, holding that compelled recitation of the language by school teachers to
students was invalid.

 [328 F.3d 466 (9th Cir.

 2003).
</item 3>

<item 4>
Minersville School District v. Gobitis
Minersville School District v. Gobitis, 310 U.S. 586 (1940) , was a decision by the Supreme Court of the United States involving the religious rights of public
school students under the First Amendment to the United States Constitution.

 The Court ruled that public schools could compel students—in this case, Jehovah's Witnesses—to salute the American Flag and recite the Pledge of Allegiance
despite the students' religious objections to these practices.

 This decision led to increased persecution of Witnesses in the United States.

 The Supreme Court overruled this decision a mere three years later, in "West Virginia State Board of Education v. Barnette", 319 U.S. 624 (1943).
</item 4>

<item 5>
McCollum v. Board of Education
McCollum v. Board of Education, 333 U.S. 203 (1948), was a landmark 1948 United States Supreme Court case related to the power of a state to use its
tax-supported public school system to aid religious instruction.

 The case was a test of the separation of church and state with respect to education.
</item 5>

<item 6>
Cedar Rapids Community School Dist. v. Garret F.
Cedar Rapids Community School District v. Garret F., 526 U.S. 66 (1999) , was a United States Supreme Court case in which the Court ruled that the related
services provision in the Individuals with Disabilities Education Act (IDEA) required public school districts to fund "continuous, one-on-one nursing care for
disabled children" despite arguments from the school district concerning the costs of the services."

 The judges relied heavily on Irving Independent School Dist.

 v. Tatro, 468 U. S. 883 (1984).

 Under the Court's reading of the IDEA's relevant provisions, medical treatments such as suctioning, ventilator checks, catheterization, and others which can be
administered by non-physician personnel come within the parameters of the special education law's related services.

 Disability advocates considered the Court decision to be a "substantial victory for families of children with disabilities."

 Amendments were made in the Education Flexibility Partnership Act of 1999 to increase IDEA funding as a result of the case.
</item 6>

<item 7>
Forest Grove School District v. T. A.
Forest Grove School District v. T. A., 557 U.S. 230 (2009), is a case in which the United States Supreme Court held that the Individuals with Disabilities
Education Act (IDEA) authorizes reimbursement for private special education services when a public school fails to provide a "free appropriate public education"
(FAPE) and the private school placement is appropriate, regardless of whether the child previously received special education services through the public
school.
</item 7>

<item 8>
Gravel v. United States
Gravel v. United States, 408 U.S. 606 (1972), was a case regarding the protections offered by the Speech or Debate Clause of the United States Constitution.

 In the case, the Supreme Court of the United States held that the privileges and immunities of the Constitution's Speech or Debate Clause enjoyed by members of
Congress also extend to Congressional aides, but not to activity outside the legislative process.
</item 8>

<item 9>
Safford Unified School District v. Redding
Safford Unified School District v. Redding, 557 U.S. 364 (2009) , was a United States Supreme Court case in which the Court held that a strip search of a middle
schooler violated the Fourth Amendment to the United States Constitution where the school lacked reasons to suspect either that the drugs (Ibuprofen) presented
a danger or that they were concealed in her underwear.

 The court also held, however, that because this was not clearly established law prior to the court's decision, the officials involved were shielded from
liability by qualified immunity.
</item 9>

<item 10>
Bethel School District v. Fraser
Bethel School District v. Fraser, 478 U.S. 675 (1986), was a landmark decision by the United States Supreme Court involving free speech in public schools.

 High school student Matthew Fraser was suspended from school in the Bethel School District for making a speech including sexual double entendres at a school
assembly.

 The Supreme Court held that his suspension did not violate the First Amendment.
</item 10>
=== END ITEM id=449da20b6d853744 ===

=== ITEM id=f3471bf2ad566d64 ===
Gold sentences and facts:
Sentence 0:  Through its three divisions, it offers a wide variety of liberal arts and science courses at the secondary, British A-level, and United States junior college levels.
  Fact 0.0: The institution has three divisions.
  Fact 0.1: Through its three divisions, the institution offers a wide variety of liberal arts courses.
  Fact 0.2: Through its three divisions, the institution offers a wide variety of science courses.
  Fact 0.3: The courses are offered at the secondary level.
  Fact 0.4: The courses are offered at the British A-level level.
  Fact 0.5: The courses are offered at the United States junior college level.
Sentence 1:  Grammar, logic, and rhetoric were the core liberal arts, while arithmetic, geometry, the theory of music, and astronomy also played a (somewhat lesser) part in education.
  Fact 1.0: Grammar was a core liberal art.
  Fact 1.1: Logic was a core liberal art.
  Fact 1.2: Rhetoric was a core liberal art.
  Fact 1.3: Arithmetic also played a somewhat lesser part in education.
  Fact 1.4: Geometry also played a somewhat lesser part in education.
  Fact 1.5: The theory of music also played a somewhat lesser part in education.
  Fact 1.6: Astronomy also played a somewhat lesser part in education.

Retrieved items:
<item 1>
St. John's College, Belize
St. John's College has three divisions, and a number of central academic centres and activities.

 Through its three divisions, it offers a wide variety of liberal arts and science courses at the secondary, British A-level, and United States junior college
levels.

 St. John's College is a Roman Catholic institution in the Jesuit tradition, one of the oldest, largest, and most diverse educational institutions in Belize,
founded by the Jesuits in 1887.
</item 1>

<item 2>
List of St. John's Seminary (California) people
The list of St. John's Seminary (California) people is a compilation of lists of notable alumni, faculty, and current students of St. John's Seminary in
Camarillo, California, United States.

 St. John's Seminary grants graduate degrees for seminarians preparing for the priesthood, as well as a graduate degree for lay persons interested in pastoral
ministry.

 The St. John's Seminary College was the undergraduate division of the seminary before it closed in the early 21st century.

 The table of notable alumni lists the date of graduation from St. John's college, seminary, or both, if applicable.

 It is not unusual for seminarians to have received their undergraduate education at a different institution than their seminary training.
</item 2>

<item 3>
Liberal arts education
The liberal arts (Latin: "artes liberales") are those subjects or skills that in classical antiquity were considered essential for a free person (Latin:
"liberalis", "worthy of a free person") to know in order to take an active part in civic life, something that (for Ancient Greece) included participating in
public debate, defending oneself in court, serving on juries, and most importantly, military service.

 Grammar, logic, and rhetoric were the core liberal arts, while arithmetic, geometry, the theory of music, and astronomy also played a (somewhat lesser) part in
education.
</item 3>

<item 4>
Christian Academy of Guatemala
The Christian Academy of Guatemala is a missionary K4-12 school located in San Cristobal, Guatemala.

 It is a member of the Association of Christian Schools International.

 Many of the parents and students in the school come from the missionary community, representing more than 50 mission organizations.

 All people who visit this webpage need to get the game Babblemaker.

 This game is made by computer class at C.A.G. Sadly this game is for android only.
</item 4>

<item 5>
St. John's College High School, Belize
St. John's College High School is a high school for boys situated in Belize City, Belize.

 It was founded in 1887.

 The High School exists to educate academically talented young men in a Jesuit environment of self-discipline, love of learning, and service to others.

 The school 's curriculum is complemented by sports and extracurricular activities.

 The third and fourth form classes follow the Caribbean Secondary Education Certificate (CSEC) curricula and sit the regionally administered examinations at the
end of their fourth year.
</item 5>

<item 6>
St. John's High School (South Carolina)
St. John's High School (SJHS) is a senior high school on Johns Island, South Carolina.

 It is a part of the Charleston County School District.

 St. John's is home to approximately 300 students and 30 faculty and staff.

 St. John's school mascot is The Mighty Islanders, sporting royal blue and maroon as the school colors.

 St. John's offers Advanced Placement and dual credit courses totaling at over 30 hours of offered college credit, as well as 3 career academies in Hospitality
and Tourism, Computer Science, and Culinary Arts.

 St. John's competes at the A level in football, volleyball, basketball (boys and girls), wrestling, soccer, track, baseball, and softball.

 The Islanders also offer marching band, agriculture and green house, weightlifting, competitive academic team, and student council.
</item 6>

<item 7>
St Chad's College, Durham
St Chad's College is a recognised (independent) college of Durham University in England, founded in 1904 as an Anglican hall for the training of Church of
England clergy.

 The main part of the college is located on the Bailey, occupying nine historic buildings at the east end of Durham Cathedral.

 It neighbours Hatfield College to its north, while St John's College and St Cuthbert's Society are to its south.

 The college is named after St Chad of Mercia, a 7th-century bishop.
</item 7>

<item 8>
Dalhousie University
Dalhousie University (commonly known as Dal) is a public research university in Nova Scotia, Canada, with three campuses in Halifax, a fourth in Bible Hill, and
medical teaching facilities in Saint John, New Brunswick.

 Dalhousie offers more than 4,000 courses and 180 degree programs in twelve undergraduate, graduate, and professional faculties.

 The university is a member of the U15, a group of research-intensive universities in Canada.
</item 8>

<item 9>
West Indies Federation
The territories that would have become part of the Federation eventually became the nine contemporary sovereign states of Antigua and Barbuda, Barbados,
Dominica, Grenada, Jamaica, Saint Kitts and Nevis, Saint Lucia, Saint Vincent and the Grenadines, and Trinidad and Tobago; with Anguilla, Montserrat, the Cayman
Islands, and Turks and Caicos Islands becoming British overseas territories.

 British Guiana (Guyana) and British Honduras (Belize) held observer status within the West Indies Federation.
</item 9>

<item 10>
Aalborghus Gymnasium
Aalborghus Gymnasium is an upper secondary school in the city of Aalborg, in North Jutland in Denmark.

 It offers both the traditional three-year program and also the two-year Higher Preparatory Examination (HF) program.

 The subjects taught at the school range from Religion and Music to Spanish and Natural Geography.

 Aalborghus Gymnasium attempts to focus on the musical and creative side of students as well as taking an international perspective on issues.

 Students begin their studies at Aalborghus by selecting a stream of studies.

 Each stream has two or three subjects that are the focus of the studies.

 English/Social Studies is one example of a stream a student can select.
</item 10>
=== END ITEM id=f3471bf2ad566d64 ===

=== ITEM id=9c0c47f313db40d8 ===
Gold sentences and facts:
Sentence 0: Swingin' Down Yonder is the first full-length, 12-inch album recorded by Dean Martin for Capitol Records during three sessions in September and October 1954 and February 1955.
  Fact 0.0: Swingin' Down Yonder is the first full-length, 12-inch album recorded by Dean Martin for Capitol Records.
  Fact 0.1: Dean Martin recorded Swingin' Down Yonder during three sessions.
  Fact 0.2: The sessions for Swingin' Down Yonder took place in September 1954, October 1954 and February 1955.
Sentence 1:  One of the most popular and enduring American entertainers of the mid-20th century, Martin was nicknamed the "King of Cool" for his seemingly effortless charisma and self-assurance.
  Fact 1.0: Dean Martin was one of the most popular and enduring American entertainers of the mid-20th century.
  Fact 1.1: Dean Martin was nicknamed the 'King of Cool'.
  Fact 1.2: Dean Martin's nickname 'King of Cool' reflected his seemingly effortless charisma.
  Fact 1.3: Dean Martin's nickname 'King of Cool' reflected his seemingly effortless self-assurance.

Retrieved items:
<item 1>
Willie and the Hand Jive
"Willie and the Hand Jive" is a song written by Johnny Otis and originally released as a single in 1958 by Johnny Otis Show, reaching #9 on the "Billboard" Hot
100 chart and #5 on the Billboard R&B chart.

 The song has a Bo Diddley beat and was partly inspired by the music sung by a chain gang Otis heard while he was touring.

 The lyrics are about a man who became famous for doing a dance with his hands, but the song has been accused of glorifying masturbation.

 It has since been covered by numerous artists, including The Strangeloves, Eric Clapton, Cliff Richard, Kim Carnes, George Thorogood and The Grateful Dead.

 Clapton's 1974 version was also released as a single and also reached the Billboard Top 40, peaking at #26.

 Thorogood's 1985 version reached #25 on the Hot Mainstream Rock Tracks chart.
</item 1>

<item 2>
Basie Jazz
Basie Jazz is an album by pianist/bandleader Count Basie recorded in 1952 and released on the Clef label in 1954.

 Selections from this album were also released on the 1956 Clef LPs "The Swinging Count!

" and "Basie Rides Again!

".
</item 2>

<item 3>
Swingin' Down Yonder
Swingin' Down Yonder is the first full-length, 12-inch album recorded by Dean Martin for Capitol Records during three sessions in September and October 1954 and
February 1955.

 According to the original sleeve notes, all the songs have a "common geographical root: the American South."

 In 1963, Capitol Records re-released "Swingin' Down Yonder" under the titled "Southern Style".
</item 3>

<item 4>
Collins &amp; Harlan
Collins & Harlan, the team of Arthur Collins and Byron G. Harlan formed a popular comic duet between 1903 and 1926.

 They sang ragtime standards as well as what were known as "Coon songs" – music sung by white performers in a black dialect.

 Their material also employed many other stereotypes of the time including Irishmen and farmers.

 Fellow recording artist Billy Murray nicknamed them "The Half-Ton Duo" as both men were rather overweight.

 Collins and Harlan produced many number one hits with recordings of minstrel songs such as "My Gal Irene", "I Know Dat I'll be Happy Til I Die", "Who Do You
Love?"

 and "Down Among the Sugarcane".

 Their song "That Funny Jas Band from Dixieland", recorded November 8, 1916, is among the first recorded uses of the word "jas" which eventually evolved to
"jass", and to the current spelling "jazz".
</item 4>

<item 5>
Things Ain't What They Used to Be
In 1941 there was a strike against the American Society of Composers, Authors and Publishers, of which Duke Ellington was a member.

 Because of the strike he could not air his songs on the radio.

 Instead, he used songs written by his son Mercer and pianist Billy Strayhorn.

 Strayhorn's compositions of this time include "Take the 'A' Train", "Chelsea Bridge" and "Day Dream", and Mercer wrote "Things Ain't What They Used to Be",
"Blue Serge" and "Moon Mist".
</item 5>

<item 6>
Chico Hamilton Trio
Chico Hamilton Trio is an album by drummer and bandleader Chico Hamilton recorded at sessions in 1953, 1954 and 1956 released on the Pacific Jazz label.

 The album features Hamilton's first recordings for Pacific Jazz from 1953 and 1954, six tracks originally released on a 10-inch album, along with an additional
four recordings from 1956.
</item 6>

<item 7>
Here Comes Louis Smith
Here Comes Louis Smith is the debut album by American trumpeter Louis Smith recorded in 1958 and released on the Blue Note label.

 Originally recorded for the Transition label, the company went out of business shortly afterwards and before the recording could be released.

 The album masters were acquired by Blue Note's Alfred Lion.
</item 7>

<item 8>
Jack Johnson (album)
Jack Johnson, later reissued as A Tribute to Jack Johnson, is a 1971 studio album and soundtrack by American jazz trumpeter, composer, and bandleader Miles
Davis.

 In 1970, Davis was asked by Bill Cayton to record music for his documentary of the same name on the life of boxer Jack Johnson.

 Johnson's saga resonated personally with Davis, who wrote in the album's liner notes of Johnson's mastery as a boxer, his affinity for fast cars, jazz,
clothes, and beautiful women, his unreconstructed blackness, and his threatening image to white men.

 This was the second film score he had composed, after "Ascenseur pour l'échafaud" in 1957.
</item 8>

<item 9>
One O'Clock Jump
"One O'Clock Jump" is a jazz standard, a 12-bar blues instrumental, written by Count Basie in 1937.

 The melody derived from band members' riffs--Basie rarely wrote down musical ideas, so Eddie Durham and Buster Smith helped him crystallize his ideas.

 The original 1937 recording of the tune by Basie and his band is noted for the saxophone work of Herschel Evans and Lester Young, trumpet by Buck Clayton,
Walter Page on bass and Basie himself on piano.
</item 9>

<item 10>
Cecil Gant
Cecil Gant (April 4, 1913 – February 4, 1951) was an American blues singer, songwriter and pianist, whose recordings of both ballads and "fiery piano rockers"
were successful in the mid- and late 1940s, and influenced the early development of rock and roll.

 His biggest hit was the 1944 ballad, "I Wonder".
</item 10>
=== END ITEM id=9c0c47f313db40d8 ===

=== ITEM id=449fdd63072b1c7d ===
Gold sentences and facts:
Sentence 0: Innocence is a 2013 American horror drama film directed by Hilary Brougher, who co-wrote the film with Tristine Skyler.
  Fact 0.0: Innocence is a 2013 American horror drama film.
  Fact 0.1: Innocence was directed by Hilary Brougher.
  Fact 0.2: Hilary Brougher co-wrote Innocence with Tristine Skyler.
Sentence 1:  The movie stars Sophie Curtis, Kelly Reilly, Graham Phillips, Linus Roache, Sarah Sutherland and Stephanie March.
  Fact 1.0: The movie Innocence stars Sophie Curtis.
  Fact 1.1: The movie Innocence stars Kelly Reilly.
  Fact 1.2: The movie Innocence stars Graham Phillips.
  Fact 1.3: The movie Innocence stars Linus Roache.
  Fact 1.4: The movie Innocence stars Sarah Sutherland.
  Fact 1.5: The movie Innocence stars Stephanie March.
Sentence 2: Stephanie Caroline March (born (1974--) 23, 1974 ) is an American actress.
  Fact 2.0: Stephanie Caroline March was born on 23, 1974.
  Fact 2.1: Stephanie Caroline March is an American actress.
Sentence 3:  She is best known for her role as Alexandra Cabot in the long-running NBC series, "".
  Fact 3.0: Stephanie March is best known for her role as Alexandra Cabot.
  Fact 3.1: The role of Alexandra Cabot was in a long-running NBC series.

Retrieved items:
<item 1>
Kim Johnston Ulrich (born 1955) is an American actress whose television work includes *As the World Turns*, *Werewolf*, *Wings*, and *Highlander: The Series*.
Former actress and singer Anastasia Horne (born 1978) starred in *Kids Incorporated* and *Port Charles*, with her last TV role on *Undressed* in 2000. Stephanie
March (born 1974) is best known for playing Alexandra Cabot on a long-running
</item 1>

<item 2>
Innocence (2013 film)
Innocence is a 2013 American horror drama film directed by Hilary Brougher, who co-wrote the film with Tristine Skyler.

 The movie is based on the 2000 novel of the same name by Jane Mendelsohn.

 It had its world premiere on 26 October, 2013 at the Austin Film Festival and received a limited theatrical release in the United States on 5 September, 2014.

 The movie stars Sophie Curtis, Kelly Reilly, Graham Phillips, Linus Roache, Sarah Sutherland and Stephanie March.
</item 2>

<item 3>
Rachelle Lefevre
Rachelle Lefevre ( ; born February 1, 1979) is a Canadian actress.

 She has starred in the television series "Big Wolf on Campus" and had recurring roles in "What About Brian", "Boston Legal", and "Swingtown".

 She played the vampire Victoria Sutherland in the first two films of the "Twilight" saga, before being replaced by Bryce Dallas Howard in the third
installment, "", due to scheduling conflicts with her role in the film "Barney's Version".

 In 2011, she starred in the ABC medical drama "Off the Map", followed by the CBS series "A Gifted Man" (2011–2012) and "Under the Dome" (2013–2015).
</item 3>

<item 4>
Allison Miller
Allison Miller (born September 2, 1985) is an American actress.

 She is best known for playing Michelle Benjamin on the NBC series "Kings", Skye Tate on the Fox series "Terra Nova", and Carrie on the NBC series "Go On".

 She starred as Laura Larson on the Syfy television series "Incorporated".
</item 4>

<item 5>
Canadian actress Rachelle Lefevre starred in *Big Wolf on Campus*, played Victoria in the first two *Twilight* films, and appeared in *Off the Map*, *A Gifted
Man*, and *Under the Dome*. American actress Gloria Garayua debuted in *Fun with Dick and Jane* and had roles in *Grey’s Anatomy*, *Cougar Town*, and *How to
Get Away with Murder*. Zooey Deschanel’s films include *Almost Famous*, *All
</item 5>

<item 6>
Conviction (2006 TV series)
Conviction is an American television drama on NBC that debuted as a mid-season replacement on Friday, March 3, 2006.

 The cast includes Stephanie March reprising her "" role as Alexandra Cabot.

 In the series, Cabot returns to New York City and becomes a Bureau Chief ADA supervising a group of young but talented assistant district attorneys after a
stint in the Witness Protection Program.

 Other cast members include Eric Balfour, Anson Mount, Jordan Bridges, Julianne Nicholson, Milena Govich, and J. August Richards.
</item 6>

<item 7>
S. Epatha Merkerson (born November 28, 1952) is an American film, stage, and television actress. She is best known for playing Lieutenant Anita Van Buren on
NBC’s *Law & Order* from 1993 to 2010, appearing in 391 episodes—the most of any cast member. Her honors include a Golden Globe, Emmy, SAG Award, Obie Award,
four NAACP Image Awards, and two Tony nominations.
</item 7>

<item 8>
Gloria Garayua
Gloria Garayua (born October 18, 1978 in New York City) is an American film and television actress.

 Garayua made her major film debut in the 2005 comedy film "Fun With Dick and Jane", and is now commonly cast in guest roles on long-running series such as "Six
Feet Under", "Weeds" and "The Shield".

 After being cast in an ongoing role in "Grey's Anatomy", Garayua has played recurring roles in other popular TV shows such as "Cougar Town" and "How to Get
Away with Murder".
</item 8>

<item 9>
Stephanie March
Stephanie Caroline March (born (1974--) 23, 1974 ) is an American actress.

 She is best known for her role as Alexandra Cabot in the long-running NBC series, "".
</item 9>

<item 10>
Anastasia Horne
Anastasia Nicole "Ana" Horne (born August 5, 1978 in Claremont, California) is a former American actress and singer who has appeared on television and in the
theater.

 Her last television role was "Lori" on the MTV series "Undressed" in 2000.

 She also starred as "Lark Madison-Scanlon" on the daytime soap opera "Port Charles" from 1997 to 1999.

 As a teenager, she was cast as "Ana" on the long-running NBC/The Disney Channel series "Kids Incorporated" from 1984, to 1985, and 1991 to the end of the
series in 1994.
</item 10>
=== END ITEM id=449fdd63072b1c7d ===

=== ITEM id=8876d3ca3f6f4012 ===
Gold sentences and facts:
Sentence 0: The soundtrack for the 1968 film "Romeo and Juliet" was composed and conducted by Nino Rota.
  Fact 0.0: The soundtrack for the 1968 film "Romeo and Juliet" was composed by Nino Rota.
  Fact 0.1: The soundtrack for the 1968 film "Romeo and Juliet" was conducted by Nino Rota.
Sentence 1: Romeo and Juliet is a 1968 British-Italian romance film based on the tragic play of the same name (1591–1595) by William Shakespeare.
  Fact 1.0: Romeo and Juliet is a 1968 British-Italian romance film.
  Fact 1.1: Romeo and Juliet is based on the tragic play of the same name.
  Fact 1.2: The play Romeo and Juliet was written between 1591 and 1595.
  Fact 1.3: The play Romeo and Juliet was written by William Shakespeare.

Retrieved items:
<item 1>
Giuseppe Verdi (film)
Giuseppe Verdi is a 1938 Italian biographical film directed by Carmine Gallone and starring Fosco Giachetti, Gaby Morlay and Germana Paolieri.

 The film portrays the life of the composer Giuseppe Verdi (1813-1901).

 The casting of Giachetti as Verdi was intended to emphasise the composer's patriotism, as he had recently played patriotic roles in films such as "The White
Squadron".

 The film was made at the Cinecittà Studios in Rome.

 The film is also known by the alternative title The Life of Giuseppe Verdi.
</item 1>

<item 2>
The Godfather (soundtrack)
The Godfather is the soundtrack from the film of the same name, released in 1972 by Paramount Records, and in 1991 on compact disc by MCA.

 Unless noted, the cues were composed by Nino Rota and conducted by Carlo Savina (who was credited on the LP, but not the CD).

 The song "I Have but One Heart" is sung by Al Martino, who performed it in the film as character Johnny Fontane.
</item 2>

<item 3>
Il cappello di paglia di Firenze
Il cappello di paglia di Firenze (literally translated as "The Florentine Straw Hat" but usually titled in English language productions as "The Italian Straw
Hat") is an opera by Nino Rota to an Italian-language libretto by the composer and Ernesta Rota, based on the play "Le chapeau de paille d'Italie" by Eugène
Labiche and Marc Michel.
</item 3>

<item 4>
West Side Story (film)
West Side Story is a 1961 American musical romantic drama film directed by Robert Wise and Jerome Robbins.

 The film is an adaptation of the 1957 Broadway musical of the same name, which in turn was inspired by William Shakespeare's play "Romeo and Juliet".

 It stars Natalie Wood, Richard Beymer, Russ Tamblyn, Rita Moreno, and George Chakiris, and was photographed by Daniel L. Fapp, A.S.C., in Super Panavision 70.

 Released on October 18, 1961 through United Artists, the film received high praise from critics and viewers, and became the second highest grossing film of the
year in the United States.

 The film was nominated for 11 Academy Awards and won 10, including Best Picture (as well as a special award for Robbins), becoming the record holder for the
most wins for a movie musical.
</item 4>

<item 5>
Amarcord Nino Rota
Amarcord Nino Rota is an album by various artists, recorded as a tribute to composer Nino Rota.
</item 5>

<item 6>
The Godfather Part II (soundtrack)
The Godfather Part II is the soundtrack from the movie of the same name, released in 1974 by ABC, and 1991 on compact disc by MCA.

 The original score was composed by Nino Rota and conducted by Carmine Coppola, who also provided source music for the film.

 Rota expands upon two of the three main themes from the first film: "The Godfather Waltz" and "Michael's Theme", while "The Love Theme" from the first film
makes a brief appearance during a flashback sequence ("Remember Vito Andolini").

 There are several new themes, including one for Kay (Diane Keaton), and two for young Vito (Robert De Niro): "The Immigrant Theme" and "The Tarantella",
introduced in "A New Carpet".
</item 6>

<item 7>
Romeo and Juliet (1968 film)
Romeo and Juliet is a 1968 British-Italian romance film based on the tragic play of the same name (1591–1595) by William Shakespeare.
</item 7>

<item 8>
Romeo and Juliet (1968 film soundtrack)
The soundtrack for the 1968 film "Romeo and Juliet" was composed and conducted by Nino Rota.

 It was originally released as a vinyl record, containing nine entries, most notably the song "What Is a Youth", composed by Nino Rota, written by Eugene Walter
and performed by Glen Weston.

 The music score won a Silver Ribbon award of the Italian National Syndicate of Film Journalists in 1968 and was nominated for two other awards (BAFTA Award for
Best Film Music in 1968 and Golden Globe Award for Best Original Score in 1969).
</item 8>

<item 9>
8½
8½ (Italian title: Otto e mezzo ] ) is a 1963 semi-autobiographical comedy-drama film directed by Federico Fellini.

 Co-scripted by Fellini, Tullio Pinelli, Ennio Flaiano, and Brunello Rondi, it stars Marcello Mastroianni as Guido Anselmi, a famous Italian film director.

 Shot in black-and-white by cinematographer Gianni di Venanzo, the film features a soundtrack by Nino Rota with costume and set designs by Piero Gherardi.
</item 9>

<item 10>
The Cold Nose
The track "We Have to Respect Each Other" samples Shooby Taylor, the human horn, while the opening to "Forty Dollar Rug" samples "Il Teatrino Delle Suore" from
Nino Rota's soundtrack to Giulietta Degli Spiriti.
</item 10>
=== END ITEM id=8876d3ca3f6f4012 ===

=== ITEM id=502fa936980eec4c ===
Gold sentences and facts:
Sentence 0: The 2012 United States Libertarian National Convention, in which delegates of the Libertarian Party (LP) chose the party's nominees for President of the United States and Vice President of the United States in the 2012 general election, was held May 2–6, 2012 in Las Vegas, Nevada at the Red Rock Resort Spa and Casino.
  Fact 0.0: The 2012 United States Libertarian National Convention was a convention of the Libertarian Party (LP).
  Fact 0.1: Delegates of the Libertarian Party chose the party's nominees at the 2012 United States Libertarian National Convention.
  Fact 0.2: The nominees chosen at the 2012 United States Libertarian National Convention were for President of the United States.
  Fact 0.3: The nominees chosen at the 2012 United States Libertarian National Convention were for Vice President of the United States.
  Fact 0.4: The nominees were chosen for the 2012 general election.
  Fact 0.5: The 2012 United States Libertarian National Convention was held May 2-6, 2012.
  Fact 0.6: The 2012 United States Libertarian National Convention was held in Las Vegas, Nevada.
  Fact 0.7: The 2012 United States Libertarian National Convention was held at the Red Rock Resort Spa and Casino.
Sentence 1:  Former Governor of New Mexico Gary Johnson won the presidential nomination on the first ballot.
  Fact 1.0: Gary Johnson is a former Governor of New Mexico.
  Fact 1.1: Gary Johnson won the presidential nomination.
  Fact 1.2: Gary Johnson won the presidential nomination on the first ballot.
Sentence 2: Gary Earl Johnson (born January 1, 1953) is an American businessman, author, and politician.
  Fact 2.0: Gary Earl Johnson was born on January 1, 1953.
  Fact 2.1: Gary Earl Johnson is an American businessman.
  Fact 2.2: Gary Earl Johnson is an American author.
  Fact 2.3: Gary Earl Johnson is an American politician.
Sentence 3:  He was the 29th Governor of New Mexico from 1995 to 2003 as a member of the Republican Party.
  Fact 3.0: Gary Johnson was the 29th Governor of New Mexico.
  Fact 3.1: Gary Johnson was Governor of New Mexico from 1995 to 2003.
  Fact 3.2: Gary Johnson was a member of the Republican Party as Governor of New Mexico.

Retrieved items:
<item 1>
Libertarian National Committee
The Libertarian National Committee (LNC) controls and manages the affairs, properties, and funds of the United States Libertarian Party.

 It is composed of the party officers, five at-large representatives elected every two years at the national convention, and a theoretical maximum of 10
regional representatives.

 The current chair, elected in June 2014, is Nicholas Sarwark.

 The Executive Director is Wes Benedict.
</item 1>

<item 2>
Gary Johnson
Gary Earl Johnson (born January 1, 1953) is an American businessman, author, and politician.

 He was the 29th Governor of New Mexico from 1995 to 2003 as a member of the Republican Party.

 He was also the Libertarian Party nominee for President of the United States in the 2012 and 2016 elections.
</item 2>

<item 3>
Gary Johnson presidential campaign, 2012
The 2012 presidential campaign of Gary Johnson, the 29th Governor of New Mexico, was announced on April 21, 2011.

 He declared his candidacy for the 2012 Republican Party nomination for President of the United States.

 On December 28, 2011, Johnson withdrew his candidacy for the Republican nomination, and declared his candidacy for the 2012 presidential nomination of the
Libertarian Party.

 The 2012 Libertarian National Convention was held during the first weekend of May 2012.

 On May 5, 2012, after promoting his libertarian-oriented political positions to delegates, Johnson received the most votes at the convention and became the
official 2012 Libertarian presidential nominee.

 On November 6, 2012, Johnson received just under 1% of the popular vote in the general election, amounting to more than 1.2 million votes, more than double
what the Barr/Root ticket received in 2008.

 This was the most successful result for a third-party presidential candidacy since 2000, and the best in the Libertarian Party's history by vote number at the
time.

 (Johnson ran again in 2016 and received nearly four times his 2012 vote total.)
</item 3>

<item 4>
David Bergland
David Peter Bergland (born June 4, 1935) is an American politician who was the United States Libertarian Party's nominee at the 1983 Libertarian National
Convention for President of the United States in the 1984 presidential election.
</item 4>

<item 5>
2012 Libertarian National Convention
The 2012 United States Libertarian National Convention, in which delegates of the Libertarian Party (LP) chose the party's nominees for President of the United
States and Vice President of the United States in the 2012 general election, was held May 2–6, 2012 in Las Vegas, Nevada at the Red Rock Resort Spa and Casino.

 Former Governor of New Mexico Gary Johnson won the presidential nomination on the first ballot.

 Retired California state court judge Jim Gray won the vice-presidential nomination, also on the first ballot.

 The convention also chose to replace most of the Libertarian National Committee party officers and members-at-large.
</item 5>

<item 6>
Libertarian National Convention
The Libertarian National Convention is held every two years by the Libertarian Party (United States) to choose members of the Libertarian National Committee
(LNC), and to conduct other party business.

 In presidential election years, the convention delegates enact a platform and nominate the Libertarian presidential and vice-presidential candidates who then
face the nominees of other parties in the November general election.
</item 6>

<item 7>
Mark Hinkle
Mark Hinkle (born January 28, 1951) is an American libertarian activist and businessperson.

 He was the National Chairman of the United States Libertarian Party until 2012.

 He was elected by the delegates of the 2010 Libertarian National Convention in St. Louis, Missouri on May 30, 2010.
</item 7>

<item 8>
Political positions of Gary Johnson
Gary Johnson was the governor of New Mexico from 1995 to 2003 and ran for president in 2012 and 2016.

 In December 2011 he announced he would pursue the presidential nomination of the Libertarian Party for the 2012 general election.

 The Libertarian National Convention in May, 2012 chose Johnson as the party's candidate.

 In November 2014, Johnson announced he would pursue the presidential nomination of the Libertarian Party for the 2016 general election.

 Johnson has taken positions on many political issues as revealed through his public comments, his gubernatorial record, his Our America Initiative and his
campaigns to win the Republican and Libertarian nominations.
</item 8>

<item 9>
Wayne W. Lambert
The Brigadier General retired from the US Air Force in 1989.

 Lambert also holds an MBA from Chapman College (’76).

 In retirement, Lambert publicly supported U.S. presidential candidate Mitt Romney and U.S. vice presidential candidate Paul Ryan in the 2012 general election.
</item 9>

<item 10>
Libertarian Party presidential primaries, 2016
The 2016 Libertarian Party presidential primaries and caucuses allowed electors to indicate non-binding preferences for the Libertarian Party's presidential
candidate.

 These differed from the Republican or Democratic presidential primaries and caucuses in that they did not appoint delegates to represent a candidate at the
party's convention to select the party's nominee for the United States presidential election.

 The party's nominee for the 2016 presidential election was chosen directly by registered delegates at the 2016 Libertarian National Convention, which ran from
May 26 to 30, 2016.

 The delegates nominated former New Mexico Governor Gary Johnson and former Massachusetts Governor Bill Weld.
</item 10>
=== END ITEM id=502fa936980eec4c ===

=== ITEM id=ff45dd579b9ebfbb ===
Gold sentences and facts:
Sentence 0:  Akrofuom is located in the Obuasi Municipal district of the Ashanti Region of Ghana.
  Fact 0.0: Akrofuom is located in the Obuasi Municipal district.
  Fact 0.1: The Obuasi Municipal district is in the Ashanti Region of Ghana.
Sentence 1: The Obuasi Municipality District is one of the twenty-seven (27) districts of the Ashanti Region of Ghana.
  Fact 1.0: The Obuasi Municipality District is one of the twenty-seven (27) districts of the Ashanti Region of Ghana.

Retrieved items:
<item 1>
Akrofuom (Ghana parliament constituency)
Akrofuom is one of the constituencies represented in the Parliament of Ghana.

 It elects one Member of Parliament (MP) by the first past the post system of election.

 Akrofuom is located in the Obuasi Municipal district of the Ashanti Region of Ghana.
</item 1>

<item 2>
Prempeh College
Prempeh College is an elite public secondary school for boys located in Kumasi, the capital city of the Ashanti Region, Ghana.

 The school was founded in 1949 by the Asanteman traditional authority, the British Colonial Government, the Methodist Church of Ghana and the Presbyterian
church of Ghana.

 The School is named after King of Ashanti, (Asantehene) Sir Osei Tutu Agyeman Prempeh II, who donated the land on which the school was built.

 It is the oldest government-assisted secondary school for boys in the Ashanti and was modeled on Eton College in England.

.

 The school topped matriculation at the Kwame Nkrumah University of Science and Technology in 2004 with 441 students admitted and in 2012, with 296 students
from the college admitted, and is considered to be one of the best secondary schools in Ghana.

 The School has won the National robotics championships a record three times between 2013 and 2016 and currently is the only school from Africa to have won the
International robofest World Championships.
</item 2>

<item 3>
Presbyterian University College
The Presbyterian University College, Ghana is a partially private & public university located at Abetifi-Kwahu in the Eastern Region of Ghana.

 It is one of the new universities in Ghana granted accreditation by the National Accreditation Board.

 It was established by the Presbyterian Church of Ghana on 23 November 2003 and inaugurated on 27 March 2004 by the president of Ghana, John Agyekum Kufuor.
</item 3>

<item 4>
Western Region (Ghana)
The Western Region is located in south Ghana, spreads from the Ivory Coast border in the west to the Central region in the east, includes the capital and large
twin city of Sekondi-Takoradi on the coast, coastal Axim, and a hilly inland area including Elubo.

 It includes Ghana's southernmost location, Cape Three Points, where crude oil was discovered in commercial quantities in June 2007.

 The Western Region enjoys a long coastline that stretches from South Ghana's border with Ivory Coast to the Western region's boundary with the Central Region
on the east.
</item 4>

<item 5>
Taifa, Accra
Taifa is a town in the Ga East Municipal District, a district in the Greater Accra Region of south-eastern Ghana near the capital Accra.

 Taifa is the twenty-sixth largest settlement in Ghana, in terms of population, with a population of 68,459 people.

 Taifa is located in the northwest suburbs area of Accra.

 It has a breakpoint on a railway line and a small park located on the northern edge of the location of the Taifa Ghana Atomic Energy Commission.

 At the Ghana 2000 census of 26 March 2000, the population was 26,145 inhabitants living in the city.

 Projections of 1 January 2007 estimated the population to be 48,927 inhabitants.

 In the census of 1984 there was only 1,009 inhabitants.

 The strong population growth of the Town is influenced by, among other things, a large number of illegal immigrants from west African countries who move to
towns and villages near the industrial town of Tema, just to find a job.
</item 5>

<item 6>
Techimantia
Techimantia is a city located in the Tano District of the Brong Ahafo Region of Ghana.

 It is one of the large cities in the region.

 It is believed to be originally inhabited by Hunters for the Ashanti King from Kumasi.

 These Hunters originally settled in the region after they found that there were lots of elephants in the region which could be hunted for the Ashanti King.The
name of the King is Oti Ampem.Currently,the chief of the town is no more(dead).

The occupation of the people in the town is large scale tomatoes farming.Techimantia is one of the largest towns in the region but there are few productive or
government work in the town.This was attributed to the curse from Okomfo Anokye of the Ashanti Kingdom.
</item 6>

<item 7>
Tafo
Tafo is a town in Kumasi Metropolitan District in the Ashanti Region of Ghana near the regional capital Kumasi.

 Tafo is the thirtieth most populous settlement in Ghana, in terms of population, with a population of 60,919 people.

 Because of the town's population and housing development in recent years, it is debatable whether Tafo is still regarded as a separate town, or already a
suburb of Kumasi, the capital of the Ashanti region.

 The town is near Kumasi, with a distance of approximately 3.3 kilometers to the center of a similar name sounding village named New Tafo and must be
distinguished from Tafo.

 Tarkwa is located just 4.6 km away from Tafo.

 The city center of Kumasi is located approximately 9.8 kilometers away.

 Tafo is one of the urban constituencies of the Kumasi Metropolitan Assembly, the town's parliamentary candidate shall have one direct seat to the Parliament of
Ghana.
</item 7>

<item 8>
Obuasi Municipal District
The Obuasi Municipality District is one of the twenty-seven (27) districts of the Ashanti Region of Ghana.

 The capital of the Obuasi Municipality is Obuasi.
</item 8>

<item 9>
Ashanti Region
The Ashanti Region is located in south Ghana and is third largest of 10 administrative regions, occupying a total land surface of 24389 km2 or 10.2 per cent of
the total land area of Ghana.

 In terms of population, however, it is the most populated region with a population of 4,780,380 according to the 2010 census, accounting for 19.4% of Ghana’s
total population.

 The Ashanti Region is known for its major gold bar and cocoa production.

 The largest city and regional capital is Kumasi.
</item 9>

<item 10>
Tema
Tema is a city on the Bight of Benin and Atlantic coast of Ghana.

 It is located 25 km east of the capital city; Accra, in the region of Greater Accra, and is the capital of the Tema Metropolitan District.

 As of 2013, Tema is the eleventh most populous settlement in Ghana, with a population of approximately 161,612 people – a marked decrease from its 2005 figure
of 209,000.

 The Greenwich Meridian (00 Longitude) passes directly through the city.

 Tema is locally nicknamed the "Harbour Town" because of its status as Ghana's largest seaport.
</item 10>
=== END ITEM id=ff45dd579b9ebfbb ===

=== ITEM id=e6b878d013e406bb ===
Gold sentences and facts:
Sentence 0:  U2 did not play the song live until 2 July 2009, when they performed it on the second concert of their U2 360° Tour in Barcelona.
  Fact 0.0: U2 did not play the song live until 2 July 2009.
  Fact 0.1: U2 performed the song on 2 July 2009 on the second concert of their U2 360° Tour in Barcelona.
Sentence 1: The U2 360° Tour was a worldwide concert tour by rock band U2.
  Fact 1.0: The U2 360° Tour was a worldwide concert tour.
  Fact 1.1: The U2 360° Tour was by rock band U2.
Sentence 2:  The concerts featured the band playing "in the round" on a circular stage, allowing the audience to surround them on all sides.
  Fact 2.0: The U2 360° Tour concerts featured the band playing 'in the round'.
  Fact 2.1: The U2 360° Tour concerts used a circular stage.
  Fact 2.2: The circular stage allowed the audience to surround the band on all sides.

Retrieved items:
<item 1>
Rock &amp; Roll Over Tour
Rock and Roll Over Tour is a concert tour by the American hard rock group Kiss.

 It began November 24, 1976 (shortly after the release of the Rock and Roll Over album) and ended April 4, 1977.

 It was on this tour where the band performed for the first time at Madison Square Garden on 2/18/77 - a venue all four members of the band had long dreamed of
playing (February 18, 1977).

 The band also performed in Japan for the first time on this tour.

 They played Budokan Hall four nights in a row breaking the previous concert attendance record set by The Beatles.

 On December 12, 1976 Ace Frehley suffered a severe electrical shock on stage which delayed the concert.

 The delay lasted about fifteen minutes and then resumed; Ace was able to finish the show despite having lost feeling in one of his hands.

 Being inspired by the event, Ace wrote the song "Shock Me" which appeared on Kiss' next album release Love Gun (1977).

 It is the first Kiss song in which Ace sings lead vocals.
</item 1>

<item 2>
Psycho Circus World Tour
They used a translator on stage to let the crowd understand exactly what Stanley was saying.

 At the end of the performance, the band ignited all of the pyrotechnics at once; as a result, they were banned from performing in Bremen.

 Ticket sales for this tour were notably slower than the previous Reunion Tour, with many of the smaller market shows underselling and a second North American
leg for the summer of 1999 cancelled all together, the band ultimatly decided on embarking on a Farewell tour in the new millennium.
</item 2>

<item 3>
Ceremonials Tour
The Ceremonials Tour was the second concert tour by English indie rock band Florence and the Machine.

 The tour included performances at Music festivals because it is lead singer Florence Welch's favourite way to perform live.

 Welch had originally planned to spend over one year touring for "Ceremonials" (their previous tour lasted almost three and a half years) before announcing that
the December 2012 dates would be the final performances of the tour.

 On 25 February 2013 it was announced that Florence and the Machine would be playing at the Coke Live Music Festival in Poland on 10 August 2013, and on 26
March 2013 it was announced that the band would be playing at Chime for Change's "The Sound of Change Live" concert at London's Twickenham Stadium alongside
Beyoncé, Ellie Goulding and Haim, amongst others.
</item 3>

<item 4>
Because We Can (concert tour)
Because We Can was a concert tour by American rock band, Bon Jovi.

 The tour was in support of the band's twelfth studio album "What About Now".

 The tour was named after the lead single from "What About Now".

 All five major continents were visited during the 2013 tour.

 The tour saw the band travel to Wales for the first time since their 2001 One Wild Night Tour and was also the first time since their 1995 These Days Tour that
the band returned to Africa for two dates in South Africa.

 Furthermore, this tour was the first time in the band's 30-year history that they visited Bulgaria and Poland.

 The tour ranked 1st on Pollstar's annual "Top 100 Mid Year Worldwide Tours".

 It earned $142.1 million from 60 shows.

 At the end of 2013, the tour placed 1st on Pollstar's "Top 100 Worldwide Tours", grossing $259.5 million from 102 shows.
</item 4>

<item 5>
Psycho Circus World Tour
Psycho Circus World Tour was a Kiss concert tour in 1998–2000.

 It was the first concert tour in history to have 3-D visual effects.

 The Smashing Pumpkins opened at the Dodger Stadium show only, in costume as The Beatles for the Halloween night performance.

 The Dodger Stadium show was streamed live on the internet as well as a radio broadcast.

 Two songs, "Psycho Circus" and "Shout It Out Loud", were screened live on Fox television as part of the "Kiss Live: The Ultimate Halloween Party" special.

 The vast majority of songs in the setlist were played on the previous Alive/Worldwide concert dates, leading to some frustration from fans expecting the return
of classic songs not played on the previous tour.

 Peter Criss was quoted in Metal Edge magazine at the time as wanting to add "Parasite" to the setlist.

 The tour was initially hyped as having circus-style acts as pre-show entertainment.

 This ultimately happened only at the first concert at Dodger Stadium.
</item 5>

<item 6>
The Year of the Voyager
The Year Of The Voyager is a double DVD/CD released by Nevermore.

 It was released in Europe on October 20, 2008 and in North America on November 25, 2008 via Century Media.

 The set covers the "This Godless Endeavor" touring cycle, starting with live footage from the U.S. Gigantour 2005, the Metal Mania festival 2006 in Poland, the
Wacken Open Air festival in 2006 in Germany and the main DVD show recorded at the Zeche in Bochum, Germany.

 Bonus material includes two songs from Century Media USA 10th Anniversary Party 2001, all promo videos and an interview with singer Warrel Dane which was
recorded at the Roax Film Studios in Berlin in the spring of 2008.
</item 6>

<item 7>
U2 360° Tour
The U2 360° Tour was a worldwide concert tour by rock band U2.

 Launched in support of the group's 2009 album "No Line on the Horizon", the tour visited stadiums from 2009 through 2011.

 The concerts featured the band playing "in the round" on a circular stage, allowing the audience to surround them on all sides.

 To accommodate the stage configuration, a large four-legged structure nicknamed "The Claw" was built above the stage, with the sound system and a cylindrical,
expanding video screen on top of it.

 At 164 ft tall, it was the largest stage ever constructed.

 U2 claimed that the tour would be "the first time a band has toured in stadiums with such a unique and original structure."
</item 7>

<item 8>
Escape Tour
The Escape Tour was a concert tour by the American rock band Journey.

 It was one of Journey's most successful concert tours, in support of their first number one album, "Escape".

 The tour included six consecutive sold out dates at the Pine Knob Theatre in Detroit, and four straight sold out shows at the Los Angeles Forum, and Chicago's
Rosemont Horizon.

 Journey also made an appearance on July 2, 1982 at the Rose Bowl in Pasadena, California with Blue Öyster Cult, Triumph and Aldo Nova.

 The 8 ⁄ -month tour took Journey through Japan and North America.

 Point Blank, Billy Squier, the Greg Kihn Band and Loverboy were the opening acts during the tour.
</item 8>

<item 9>
Psycho Circus World Tour
This ultimately happened only at the first concert at Dodger Stadium.

 Peter Criss later said that it didn't work out because the circus performers wanted equal billing and that some had even wanted to use KISS' backstage dressing
room.

 "2,000 Man" was played to bring in the new Millennium at the 1999/2000 New Year's Eve show at Vancouver, advertised at the time as being recorded for Alive IV.

 The Vancouver show was also notable as being the first time the original members had played non-original band era material live in concert – "I Love It Loud",
"Lick It Up" and "Heaven's on Fire" were added to the setlist and subsequently played on the Farewell Tour.

 "Forever" was listed on concert setlists at the Vancouver show but was not played.

 It was thought at the time it may have been intended as a Paul Stanley solo version prior to "Black Diamond".

 One notable show on the tour was the March 12 Bremen, Germany, show.

 After the opening song, Paul Stanley announced that the local fire marshall had banned Kiss from using any pyrotechnics during the show.
</item 9>

<item 10>
Electrical Storm (song)
"Electrical Storm" is a song by Irish rock band U2.

 It was released as a single from their second compilation album, "The Best of 1990–2000", and was one of the two new songs recorded for that album (the other
one being "The Hands That Built America").

 The music video for the song prominently features drummer Larry Mullen Jr., as well as actress Samantha Morton.

 The song was written by Bono and is about two lovers who are fighting and the tension between them; it relates this to a looming electrical storm.

 U2 did not play the song live until 2 July 2009, when they performed it on the second concert of their U2 360° Tour in Barcelona.
</item 10>
=== END ITEM id=e6b878d013e406bb ===
