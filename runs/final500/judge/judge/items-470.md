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

This file has 12 items.

=== ITEM id=6705e77af7ff1b6c ===
Gold sentences and facts:
Sentence 0: Ex parte Young, 209 U.S. 123 (1908) , is a United States Supreme Court case that allows suits in federal courts against officials acting on behalf of states of the union to proceed despite the State's Sovereign immunity, when the State acted unconstitutionally.
  Fact 0.0: Ex parte Young is a United States Supreme Court case.
  Fact 0.1: Ex parte Young is cited as 209 U.S. 123.
  Fact 0.2: Ex parte Young is dated 1908.
  Fact 0.3: Ex parte Young allows suits in federal courts against officials acting on behalf of states of the union.
  Fact 0.4: Ex parte Young allows such suits to proceed despite the state's sovereign immunity.
  Fact 0.5: Ex parte Young allows such suits when the state acted unconstitutionally.
Sentence 1:  The case was a test of the separation of church and state with respect to education.
  Fact 1.0: The Ex parte Young case was a test of the separation of church and state.
  Fact 1.1: The test of church and state separation in the Ex parte Young case concerned education.

Retrieved items:
<item 1>
Idaho v. Coeur d'Alene Tribe of Idaho
Idaho v. Coeur d'Alene Tribe of Idaho, 521 U.S. 261 (1997), was a United States Supreme Court case in which the Court held that the Coeur d'Alene Tribe could
not maintain an action against the state of Idaho to press its claim to Lake Coeur d'Alene due to the state's Eleventh Amendment immunity from suit,
notwithstanding the exception recognized in "Ex parte Young".

 The case was an important precedent for aboriginal title in the United States and sovereign immunity in the United States.
</item 1>

<item 2>
Owasso Independent School District v. Falvo
Owasso Independent School District v. Falvo, 534 U.S. 426 (2002) , was a case in which the United States Supreme Court held in favor of the school district that
students scoring each other's tests and calling out the grades do not violate the Family Educational Rights and Privacy Act of 1974 (FERPA).

 Justice Kennedy wrote the opinion for the unanimous court.

 Justice Scalia wrote a concurring opinion in which he agreed with the ruling, but took issue with parts of Kennedy's opinion.
</item 2>

<item 3>
Dannel McCollum
Dannel McCollum was the mayor of Champaign, Illinois from 1987-1999.

 He is the author of the book "The Lord Was Not on Trial" about his mother Vashti McCollum's landmark 1948 Supreme Court case McCollum v. Board of Education,
which struck down religious education in public schools.

 He also wrote the book "Remembering Champaign County".
</item 3>

<item 4>
Duncan v. Kahanamoku
Duncan v. Kahanamoku, 327 U.S. 304 (1946), was a decision by the United States Supreme Court.

 It is often associated with the Japanese exclusion cases ("Hirabayashi v. United States", "Korematsu v. United States" and "Ex parte Endo") because it involved
wartime curtailment of fundamental civil liberties under the aegis of military authority, though in this case neither the plaintiff nor the nominal defendant
were Japanese.
</item 4>

<item 5>
Seminole Tribe of Florida v. Florida
Seminole Tribe of Florida v. Florida, 517 U.S. 44 (1996) , was a United States Supreme Court case which held that Article One of the U.S. Constitution did not
give the United States Congress the power to abrogate the sovereign immunity of the states that is further protected under the Eleventh Amendment.

 Such abrogation is permitted where it is necessary to enforce the rights of citizens guaranteed under the Fourteenth Amendment as per "Fitzpatrick v. Bitzer"
427 U.S. 445 (1976).

 The case also held that the doctrine of "Ex parte Young", 209 U.S. 123 (1908), which allows state officials to be sued in their official capacity for
prospective injunctive relief, was inapplicable under these circumstances, because any remedy was limited to the one that Congress had provided.
</item 5>

<item 6>
Abington School District v. Schempp
Abington School District v. Schempp, 374 U.S. 203 (1963) , was a United States Supreme Court case in which the Court decided 8–1 in favor of the respondent,
Edward Schempp, and declared school-sponsored Bible reading in public schools in the United States to be unconstitutional.

 The Chief Justice of the Supreme Court during this case was Earl Warren.
</item 6>

<item 7>
McCollum v. Board of Education
McCollum v. Board of Education, 333 U.S. 203 (1948), was a landmark 1948 United States Supreme Court case related to the power of a state to use its
tax-supported public school system to aid religious instruction.

 The case was a test of the separation of church and state with respect to education.
</item 7>

<item 8>
Ex parte Endo
Ex parte Endo, or Ex parte Mitsuye Endo, 323 U.S. 283 (1944), was a United States Supreme Court ex parte decision handed down on December 18, 1944, in which the
Justices unanimously ruled that the U.S. government could not continue to detain a citizen who was "concededly loyal" to the United States.

 Although the Court did not touch on the constitutionality of the exclusion of people of Japanese ancestry from the West Coast—which they had found not to
violate citizen rights in their "Korematsu v. United States" decision on the same date—the Endo ruling nonetheless led to the reopening of the West Coast to
Japanese Americans after their incarceration in camps across the U.S. interior during World War II.
</item 8>

<item 9>
Criminal law in the Waite Court
Notable cases include "United States v. Cruikshank" (1875), "United States v. Reese" (1875), "Reynolds v. United States" (1878), "Wilkerson v. Utah" (1879), the
"Trade-Mark Cases" (1879), "Strauder v. West Virginia" (1880), "Pace v. Alabama" (1883), "United States v. Harris" (1883), "Ex parte Crow Dog" (1883), "Hurtado
v. California" (1884), "Clawson v. United States" (1885), "Yick Wo v. Hopkins" (1886), "United States v. Kagama" (1886), "Ker v. Illinois" (1886), and "Mugler
v. Kansas" (1887).
</item 9>

<item 10>
Ex parte Young
Ex parte Young, 209 U.S. 123 (1908) , is a United States Supreme Court case that allows suits in federal courts against officials acting on behalf of states of
the union to proceed despite the State's Sovereign immunity, when the State acted unconstitutionally.
</item 10>
=== END ITEM id=6705e77af7ff1b6c ===

=== ITEM id=840c2e86ed614128 ===
Gold sentences and facts:
Sentence 0: St. Elizabeths Hospital opened in 1855 as the first federally operated psychiatric hospital in the United States.
  Fact 0.0: St. Elizabeths Hospital opened in 1855.
  Fact 0.1: St. Elizabeths Hospital opened as the first federally operated psychiatric hospital in the United States.
Sentence 1:  Founded in 1861, it is the longest continuously operating hospital in the District.
  Fact 1.0: St. Elizabeths Hospital was founded in 1861.
  Fact 1.1: St. Elizabeths Hospital is the longest continuously operating hospital in the District.

Retrieved items:
<item 1>
Vassar Brothers Medical Center
Vassar Brother Medical Center (locally known as Vassar Hospital or VBMC) is a major medical facility located in the city of Poughkeepsie, New York that is a
member of the Health Quest network, a nonprofit family of hospitals and healthcare centers in the Hudson Valley area.

 VBMC is one of two major medical centers located in Dutchess County, New York, the other being Mid-Hudson Regional Hospital of Westchester Medical Center.

 The hospital was first incorporated in 1882 under the name Vassar Brothers Hospital, following contributions made by John Guy Vassar following the death of his
brother.

 The hospital opened its doors in 1887, and initially had 40 beds divided up into four wards, each containing ten beds.

 At the time the hospital opened it also contained a labor and delivery ward, a children's ward, a nursery, three private rooms, and two isolation rooms.

 Several major renovations occurred in 1983, 1991, and 2001; adding critical care areas, a new operating and delivery wing, and a cancer care center,
respectively.
</item 1>

<item 2>
Montefiore New Rochelle Hospital
Montefiore New Rochelle Hospital (formerly Sound Shore Medical Center) is a community-based, teaching hospital located at 16 Guion Place in the city of New
Rochelle, in Westchester County, New York, and affiliated with the Albert Einstein College of Medicine.

 The hospital opened in 1892 as New Rochelle Hospital.

 On November 6, 2013, Sound Shore was acquired by the Bronx-based Montefiore Medical Center, the University Hospital for the Albert Einstein College of
Medicine, was renamed Montefiore New Rochelle Hospital, and became part of the Montefiore Health System.
</item 2>

<item 3>
Providence Hospital (Washington, D.C.)
Providence Hospital is a 408 bed hospital located in the District of Columbia.

 Founded in 1861, it is the longest continuously operating hospital in the District.

 Providence Hospital is a member of Ascension Health, the largest non-profit health care organization in the United States.
</item 3>

<item 4>
St. John's Regional Medical Center (California)
St. John's Regional Medical Center is a hospital located in Oxnard, California in the United States, and is operated by Dignity Health, along with its sister
hospital, St. John's Pleasant Valley Hospital in Camarillo, California.

 The hospital was founded in 1912.

 St. John's Regional Medical Center and St. John's Pleasant Valley Hospital offer comprehensive medical services, including 24-hour emergency medical and
surgical services and care, cancer and oncology care and support, cardiovascular care, community outreach and screenings, diagnostic imaging services,
laboratory services, maternity and women's services, neonatal intensive care, palliative care, patient and family education, rehabilitation services, spine and
orthopedic care, weight loss surgery, wound healing and oxygen therapy, and more.

 Together, St. John's Regional Medical Center and St. John's Pleasant Valley Hospital represent the largest acute-care health organization in Ventura County.

 St. John's hospitals serve all of Ventura County and beyond, including the cities of Camarillo, Moorpark, Oxnard, Port Hueneme, Ventura, and Somis.
</item 4>

<item 5>
Douglass (Washington, D.C.)
Douglass is a residential neighborhood in Southeast Washington, D.C., on the eastern side of St. Elizabeths Hospital, on the border of the Congress Heights
Metro Station.

 It is bounded by Suitland Parkway to the north and east, Alabama Avenue to the south, and the St. Elizabeths campus to the west.
</item 5>

<item 6>
History of psychiatric hospitals in Pittsburgh
Pittsburgh, Pennsylvania, United States, is home to a plethora of hospitals, be it for the physically ill or, what was defined as the insane and feeble minded.

 In 1891, the St. Margaret Memorial Hospital opened under John H. Shoenberger.

 In 1908, the Pittsburgh Hospital was run under C. A. Fagan, and on June 1, 1889, the South Side Hospital opened under Anna M. Rindlaub.

 Pittsburgh, commonly known as the "Steel City," also built up a reputation for its abundance of hospitals that had opened up within a fairly close time period
of each other.

 During this time, many new mental hospitals were established due to overcrowding.

 In 1899, a building to house the mentally insane, Mayview State Hospital, was constructed; but it was not long before the complex had a reputation “as a place
of sorrow” due to the lack of proper health care patients received during their stay.
</item 6>

<item 7>
St. Elizabeths Hospital
St. Elizabeths Hospital opened in 1855 as the first federally operated psychiatric hospital in the United States.

 Housing over 8,000 patients at its peak in the 1950s, the hospital at one point had a fully functioning medical-surgical unit, a school of nursing, and
accredited internships and psychiatric residencies.

 Its campus was designated a National Historic Landmark in 1990.
</item 7>

<item 8>
Churches and related institutions: Karlsøy’s 1854 wooden church seats 570 and holds occasional services; Manhattan’s Jesuit St. Ignatius parish began in 1851
and was renamed in 1898. Istanbul’s Spring monastery is an ancient Orthodox pilgrimage shrine with an 1835 church; Hagia Irene is an Orthodox museum and concert
hall. Ballston Spa’s St. Mary’s is the Albany Diocese’s fourth-oldest parish; Iowa City’s church and rect
</item 8>

<item 9>
Vassar Brothers Medical Center
In 2002 Vassar Brothers Hospital officially became Vassar Brothers Medical Center.

 In September 2016, VBMC announced a new $500 million expansion which would almost double the number of available rooms.

 Construction on the expansion is expected to be completed in 2019.
</item 9>

<item 10>
Heritage Hospital
Heritage Hospital is a hospital located in Tarboro, North Carolina.

 It is a part of the University Health Systems of Eastern Carolina (UHSEC).

 Edgecombe General Hospital opened as a county-owned hospital in 1916.

 It succeeded Pittman Hospital, which opened in 1901.

 In 1959, the Hill-Burton Act helped combine Edgecombe General Hospital, with three other facilities.

 Edgecombe County sold the hospital to Hospital Corporation of America (HCA) in 1982.

 HCA opened a 127-bed facility in 1985, named Heritage Hospital.

 UHSEC bought Heritage Hospital in 1998 from HCA.

 The hospitals focus is as a community hospital.

 The hospital has 101 general and 16 rehabilitation hospital beds.

 It has five Shared Inpatient/Ambulatory Surgery, two Endoscopy, and one C-Section operating rooms.
</item 10>
=== END ITEM id=840c2e86ed614128 ===

=== ITEM id=dc05e9203aa3fdab ===
Gold sentences and facts:
Sentence 0:  At the 2002 Commonwealth Games she won a silver medal with Sara Petersen in the women's doubles, and a bronze medal in the mixed team.
  Fact 0.0: She won a silver medal at the 2002 Commonwealth Games.
  Fact 0.1: She won the silver medal with Sara Petersen in the women's doubles at the 2002 Commonwealth Games.
  Fact 0.2: She won a bronze medal in the mixed team at the 2002 Commonwealth Games.
Sentence 1: The 2002 Commonwealth Games, officially known as the XVII Commonwealth Games and commonly known as Manchester 2002 were held in Manchester, England, from 25 July to 4 August 2002.
  Fact 1.0: The 2002 Commonwealth Games were officially known as the XVII Commonwealth Games.
  Fact 1.1: The 2002 Commonwealth Games were commonly known as Manchester 2002.
  Fact 1.2: The 2002 Commonwealth Games were held in Manchester, England.
  Fact 1.3: The 2002 Commonwealth Games were held from 25 July to 4 August 2002.

Retrieved items:
<item 1>
2014 Rogers Cup – Women's Doubles
Jelena Janković and Katarina Srebotnik were the defending champions, but they chose not to compete together.

 Janković played alongside Klára Koukalová, but lost in the second round to Michaëlla Krajicek and Barbora Záhlavová-Strýcová.

 Srebotnik teamed up with Květa Peschke, but lost in the second round to Chan Hao-ching and Chan Yung-jan.

<br>
</item 1>

<item 2>
2007 Kremlin Cup – Women's Doubles
Květa Peschke and Francesca Schiavone were the defending champions, but Schiavone chose not to participate, and only Peschke competed that year.
</item 2>

<item 3>
Nicole Gordon (badminton)
Nicole Gordon (born 17 March 1976 in Lower Hutt, New Zealand) is a female badminton player from New Zealand.

 At the 2002 Commonwealth Games she won a silver medal with Sara Petersen in the women's doubles, and a bronze medal in the mixed team.
</item 3>

<item 4>
2010 Toray Pan Pacific Open – Doubles
Alisa Kleybanova and Francesca Schiavone were the defending champions.

 Both were present, but competed with different partners.Kleybanova competed with Hsieh Su-wei, but lost in the first round to Benešová and Záhlavová-Strýcová,
while Schiavone competed with Tathiana Garbin, however withdrew before their quarterfinal match against Lisa Raymond and Rennae Stubbs.
</item 4>

<item 5>
2005 Challenge Bell – Doubles
Carly Gullickson and María Emilia Salerni were the defending champions, but Gullickson decided not to participate this year.

 Salerni partnered with Marion Bartoli, but withdrew from their semifinal match against Līga Dekmeijere and Ashley Harkleroad.
</item 5>

<item 6>
Graph query row:
event: 2002 Commonwealth Games
location: Manchester
</item 6>

<item 7>
1999 Egger Tennis Festival – Doubles
The 1999 Egger Tennis Festival – Singles was the singles event of the twenty-ninth edition of the Egger Tennis Festival; a WTA Tier IV tournament and the second
most prestigious women's tennis tournament held in Austria.

 Laura Montalvo and Paola Suárez were the defending champions, but they didn't compete together this year.

 Montalvo played with Olga Lugina as the third seed, while Suárez teamed up with Virginia Ruano Pascual as the second seed.
</item 7>

<item 8>
2010 US Open – Mixed Doubles
Carly Gullickson and Travis Parrott were the defending champions, but lost in the first round to Gisela Dulko and Pablo Cuevas 2–6, 4–6.
</item 8>

<item 9>
1999 Copa Colsanitas – Doubles
The 1999 Copa Colsanitas – Doubles was the doubles event of the second edition of the Copa Colsanitas; a WTA Tier IV tournament and the most prestigious women's
tennis tournament held in Colombia and Hispanic America.

 Janette Husárová and Paola Suárez were the defending champions but only Suárez competed that year with Laura Montalvo.
</item 9>

<item 10>
Nicole Strausak
Nicole Strausak (born October 21, 1971) is a Swiss curler who retired from competitive curling in March 2005.

 Amongst other awards, she received a World Junior Silver medal in 1991, a European Silver medal in 1993 and a World Silver medal in 2000.

 In 1999, at the European Curling Championships in Chamonix, as well as receiving a Bronze medal, she was awarded the Fair Play & Friendly Award by her fellow
athletes for her conduct and attitude, on and off the ice.
</item 10>
=== END ITEM id=dc05e9203aa3fdab ===

=== ITEM id=301e412e7bcce78d ===
Gold sentences and facts:
Sentence 0:  He received an excellent education in both Arabic and Greek, having studied Arabic grammar with the famous poet Abul ʿAla Al-Maʿarri.
  Fact 0.0: He received an excellent education in Arabic.
  Fact 0.1: He received an excellent education in Greek.
  Fact 0.2: He studied Arabic grammar with the famous poet Abul ʿAla Al-Maʿarri.
Sentence 1: Abul ʿAla Al-Maʿarri (Arabic أبو العلاء المعري "Abū al-ʿAlāʾ al-Maʿarrī ", full name أبو العلاء أحمد بن عبد الله بن سليمان التنوخي المعري "Abū al-ʿAlāʾ Aḥmad ibn ʿAbd Allāh ibn Sulaimān al-Tanūẖī al-Maʿarrī "; December 973 – May 1057) was a blind Arab philosopher, poet, and writer.
  Fact 1.0: Abul ʿAla Al-Maʿarri's Arabic name is أبو العلاء المعري (Abū al-ʿAlāʾ al-Maʿarrī).
  Fact 1.1: Abul ʿAla Al-Maʿarri's full name is أبو العلاء أحمد بن عبد الله بن سليمان التنوخي المعري (Abū al-ʿAlāʾ Aḥmad ibn ʿAbd Allāh ibn Sulaimān al-Tanūẖī al-Maʿarrī).
  Fact 1.2: Abul ʿAla Al-Maʿarri was born in December 973.
  Fact 1.3: Abul ʿAla Al-Maʿarri died in May 1057.
  Fact 1.4: Abul ʿAla Al-Maʿarri was blind.
  Fact 1.5: Abul ʿAla Al-Maʿarri was an Arab philosopher.
  Fact 1.6: Abul ʿAla Al-Maʿarri was an Arab poet.
  Fact 1.7: Abul ʿAla Al-Maʿarri was an Arab writer.

Retrieved items:
<item 1>
Antarah ibn Shaddad
Antarah ibn Shaddad (Arabic: عنترة بن شداد العبسي‎ ‎ , "ʿAntarah ibn Shaddād al-ʿAbsī";  525–608), also known as ʿAntar, was a pre-Islamic Arab knight and poet,
famous for both his poetry and his adventurous life.

 His chief poem forms part of the "Mu'allaqāt", the collection of seven "hanging odes" legendarily said to have been suspended in the Kaaba.

 The account of his life forms the basis of a long and extravagant romance.
</item 1>

<item 2>
Al-Mutanabbi
Al-Mutanabbi had a great pride in himself through his poetry.

 Among the topics he discussed were courage, the philosophy of life, and the description of battles.

 Many of his poems were and still are widely spread in today's Arab world and are considered to be proverbial.
</item 2>

<item 3>
Muhammad ibn Ali ibn Abdallah
Muhammad ibn Ali ibn Abdallah or Muhammad al-Imâm was the son of Ali ibn Abd Allah ibn al-Abbas and great-grandson of al-‘Abbas ibn ‘Abd al-Muttalib, the uncle
of the Islamic prophet, Muhammad.

 He was born in Humaima in Palestine, and was the progenitor of the Abbasid dynasty.
</item 3>

<item 4>
Da'ud Abu al-Fadl
Da'ud Abu al-Fadl (1161–1242) was a Karaite Jewish physician who lived in the Ayyubid Egypt in the twelfth century CE.

 He born in Cairo in 1161 and died there about 1242.

 Having studied medicine under the Jewish physician Hibat Allah ibn Jami, and under Abu al-Fafa'il ibn Naqid, he became the court physician of the sultan
al-Malik al-'Adil Abu Bakr ibn Ayyub, the brother and successor of Saladin.

 He was also chief professor at the al-Nasiri Hospital at Cairo, where he had a great many pupils, among them being the historian ibn Abi Usaibiyyah.

 The latter declared that Abu al-Fadl was the most skillful physician of the time and that his success in curing the sick was miraculous.

 Abu al-Fadl was the author of an Arabic pharmacopoeia in twelve chapters, entitled "Aḳrabadhin", and treating chiefly of antidotes.
</item 4>

<item 5>
Abd Allah ibn Abbas
Abd Allah ibn Abbas (Arabic: عبد الله ابن عباس‎ ‎ ) or ′Abd Allah ibn al-′Abbas otherwise called (Ibn Abbas; Al-Habr; Al-Bahr; The Doctor; The Sea) was born c.
619 CE.

 He was the son of Al-‘Abbas ibn ‘Abd al-Muttalib, an uncle of the Islamic prophet Muhammad, and a nephew of the Maymunah bint al-Harith, who later became
Muhammad's wife.

 He was one of Muhammad's cousins and one of the early Qur'an scholars.
</item 5>

<item 6>
Ibn al-Khatib
Lisan ad-Din ibn al-Khatib (Born 16 November 1313, Loja– died 1374, Fes, Morocco) (Full name Muhammad ibn Abd Allah ibn Said ibn Ali ibn Ahmad al-Salmani) was a
Moroccan Arab polymath poet, writer, historian, philosopher, physician and politician from Emirate of Granada.

 Some of his poems decorate the walls of the Alhambra in Granada.
</item 6>

<item 7>
Shaykh Muhammad ibn Thaleb
Muhammad ibn Thalib ibn Abd Allah ibn Ni`mat Allah ibn Sadr ad-Din ibn Shaykh Baha' ad-Din ash-Shirazi (Arabic: محمد ابن طالب عبدالله ابن نعمات الله ابن صدر
الدين ابن شيخ بهاءالدين الشيرازي‎ ‎ ) was a 15th-century Persian physician from Shiraz, Iran.
</item 7>

<item 8>
Al-Maʿarri
Abul ʿAla Al-Maʿarri (Arabic أبو العلاء المعري "Abū al-ʿAlāʾ al-Maʿarrī ", full name أبو العلاء أحمد بن عبد الله بن سليمان التنوخي المعري "Abū al-ʿAlāʾ Aḥmad
ibn ʿAbd Allāh ibn Sulaimān al-Tanūẖī al-Maʿarrī "; December 973 – May 1057) was a blind Arab philosopher, poet, and writer.

 Al-Maʿarri held and expressed an irreligious world view which was met with controversy, but in spite of it, he is regarded as one of the greatest classical
Arabic poets.
</item 8>

<item 9>
Abd Allah ibn al-Fadl
Little is known of his life, apart from what can be gleaned from manuscripts of his texts.

 He was a deacon and the grandson of a bishop.

 He received an excellent education in both Arabic and Greek, having studied Arabic grammar with the famous poet Abul ʿAla Al-Maʿarri.

 Several of his works and translations were commissioned by notables from Antioch and neighbouring cities in Muslim territory.
</item 9>

<item 10>
Abdal
Abdāl ("lit": substitutes, but which can also mean "generous" ["karīm"] and "noble" ["sharīf"]) is a term used in Islamic metaphysics and Islamic mysticism,
both Sunni and Shiite, to refer to a particularly important group of God's saints.

 In the tradition of Sunni Islam in particular, the concept attained an especially important position in the writings of the Sunni mystics and theologians,
whence it appears in the works of Sunni authorities as diverse as Abu Talib al-Makki (d. 956), Ali Hujwiri (d. 1072), Ibn Asakir (d. 1076), Khwaja Abdullah
Ansari (d. 1088), and Ibn Khaldun (d. 1406).
</item 10>
=== END ITEM id=301e412e7bcce78d ===

=== ITEM id=30b50d255269a6d3 ===
Gold sentences and facts:
Sentence 0: Tritonia (flame freesia) is a genus of flowering plants in the iris family first described as a genus in 1802.
  Fact 0.0: Tritonia, also called flame freesia, is a genus of flowering plants.
  Fact 0.1: Tritonia belongs to the iris family.
  Fact 0.2: Tritonia was first described as a genus in 1802.
Sentence 1: Phyteuma is a genus of flowering plants in the family Campanulaceae, native to Europe and Morocco.
  Fact 1.0: Phyteuma is a genus of flowering plants.
  Fact 1.1: Phyteuma is in the family Campanulaceae.
  Fact 1.2: Phyteuma is native to Europe.
  Fact 1.3: Phyteuma is native to Morocco.

Retrieved items:
<item 1>
Phyteuma
Phyteuma is a genus of flowering plants in the family Campanulaceae, native to Europe and Morocco.
</item 1>

<item 2>
Zygotritonia
Zygotritonia is a genus of herbaceous, perennial and bulbous plants in the Iris family (Iridaceae).

 It contains four species distributed in Tropical Africa.

 The genus name is derived from the word zygomorphic (meaning bilabiate), and the apparent resemblance to some species in the genus "Tritonia".
</item 2>

<item 3>
Tritoniopsis
Tritoniopsis is a genus of flowering plants in the family Iridaceae, first described as a genus in 1926.

 The entire genus is endemic to Cape Province in South Africa.

 The genus name refers to the African genus "Tritonia" and is combined with the Greek word "opsis", meaning "look-alike".
</item 3>

<item 4>
Thonningia
Thonningia is a monotypic genus of flowering plants in the family Balanophoraceae containing the single species Thonningia sanguinea.

 It is distributed throughout much of southern and western Africa, particularly the tropical regions.

 Common names for the plant include ground pineapple.

 A familiar plant to humans, it has an extremely long list of common names in many African languages.

 Many names are inspired by the resemblance of the plant's inflorescence to a pineapple or palm tree.

 Some of the names can be translated as pineapple of the bush (from Anyi), duiker's kolanut (from Igala), and crown of the ground (from Yoruba).
</item 4>

<item 5>
Conophytum
Conophytum is a genus of South African and Namibian succulent plants that belong to the Aizoaceae family.

 The name is derived from the Latin “conus” (cone) and Greek “phytum” (plant).

 The plants are also known as knopies (buttons), waterblasies (water blisters), sphaeroids, conos, cone plants, dumplings, or button plants.
</item 5>

<item 6>
Tritonia (plant)
Tritonia (flame freesia) is a genus of flowering plants in the iris family first described as a genus in 1802.

 They are naturally distributed across southern Africa, with a high concentration of species in Cape Province of western South Africa.

 The genus is closely related to the genus "Ixia".
</item 6>

<item 7>
Vascular plant
Vascular plants (from Latin "vasculum": duct), also known as tracheophytes (from the equivalent Greek term "trachea") and also higher plants, form a large group
of plants (c. 308,312 accepted known species ) that are defined as those land plants that have lignified tissues (the xylem) for conducting water and minerals
throughout the plant.

 They also have a specialized non-lignified tissue (the phloem) to conduct products of photosynthesis.

 Vascular plants include the clubmosses, horsetails, ferns, gymnosperms (including conifers) and angiosperms (flowering plants).

 Scientific names for the group include Tracheophyta and Tracheobionta.
</item 7>

<item 8>
Tricyrtis
Tricyrtis is a genus of Asian flowering plants in the lily family, with approximately 20 known species.

 The species are commonly known in English as toad lilies.

 The genus has a native range from the Himalayas to eastern Asia, including China, Japan, Philippines and Taiwan, and a few species are cultivated for their
ornamental qualities in other parts of the world.
</item 8>

<item 9>
Cyrtanthus
Cyrtanthus is a genus of perennial, herbaceous and bulbous plants in the family Amaryllidaceae, subfamily Amaryllidoideae.
</item 9>

<item 10>
Alismataceae
The water-plantains (Alismataceae) are a family of flowering plants, comprising 11 genera and between 85 and 95 species.

 The family has a cosmopolitan distribution, with the greatest number of species in temperate regions of the Northern Hemisphere.

 Most of the species are herbaceous aquatic plants growing in marshes and ponds.
</item 10>
=== END ITEM id=30b50d255269a6d3 ===

=== ITEM id=2e7ce0f02d2f6f1a ===
Gold sentences and facts:
Sentence 0:  Պողոսյան , born July 15, 1990) is an American professional ice hockey defenseman currently playing for the Buffalo Sabres.
  Fact 0.0: Պողոսյան was born on July 15, 1990.
  Fact 0.1: Պողոսյան is an American professional ice hockey defenseman.
  Fact 0.2: Պողոսյան is currently playing for the Buffalo Sabres.
Sentence 1: Daron Vartan Malakian (Armenian: Տարօն/Տարոն Վարդան Մալաքեան, born July 18, 1975) is an Armenian–American singer-songwriter, multi-instrumentalist, and record producer.
  Fact 1.0: The Armenian name of Daron Vartan Malakian is Տարօն/Տարոն Վարդան Մալաքեան.
  Fact 1.1: Daron Vartan Malakian was born on July 18, 1975.
  Fact 1.2: Daron Vartan Malakian is an Armenian-American singer-songwriter.
  Fact 1.3: Daron Vartan Malakian is a multi-instrumentalist.
  Fact 1.4: Daron Vartan Malakian is a record producer.

Retrieved items:
<item 1>
Bad Acid Trip
Bad Acid Trip is an American metal band, currently signed to System of a Down lead singer Serj Tankian's record label, Serjical Strike Records.

 Their 2004 album, "Lynch the Weirdo", was produced by Daron Malakian, another System of a Down member.

 In 2006 they performed on the second stage at the Ozzfest summer tour.

 The band has three full-length releases ("For the Weird by the Weird", "Lynch the Weirdo", and "Humanly Possible") and multiple EPs.
</item 1>

<item 2>
Daron Jones
Daron Tavaris Jones (born December 27, 1976) is an American singer-songwriter, record producer.

 He is a member in Bad Boy Records group 112.

 His debut solo album titled "Uncensored" was released December 5, 2010.

 The first single from the album is the digitally-released "Money."

 Daron digitally released "Christmas in Atlanta" in December 2014, which reached the top 30 on the iTunes R&B chart.

 Daron is currently in the studio working on new music.
</item 2>

<item 3>
Jimmy Chamberlin
James Joseph Chamberlin (born June 10, 1964) is an American drummer and record producer.

 He is best known as the drummer for the alternative rock band The Smashing Pumpkins.

 Following the 2000 breakup of the band, Chamberlin joined Pumpkins frontman Billy Corgan in the supergroup Zwan and also formed his own group, the Jimmy
Chamberlin Complex.

 In 2005, Chamberlin joined Corgan in reforming The Smashing Pumpkins; he eventually left the group in March 2009, though he returned again in 2015 for a summer
tour.

 He performed in the group Skysaw until 2012.

 He is currently active under the Jimmy Chamberlin Complex name.

 In addition to his current work as CEO, Chamberlin has joined Chicago jazz saxophonist Frank Catalano for a string of 2013–15 performances in the Chicago area.

 An EP by Catalano and Chamberlin "Love Supreme Collective - EP" was released on the 29th of July 2014.
</item 3>

<item 4>
Doug Grean
Doug Grean is an American record producer, audio engineer, guitarist, based in Los Angeles.

 He has worked with artists Sheryl Crow, Ricki Lee Jones, Glen Campbell, Scott Weiland, Slash, Velvet Revolver, Stone Temple Pilots, The Crystal Method, Tim
McGraw, Cyndi Lauper, Annabella Lwin (Bow Wow Wow), Corey Glover (Living Color), Camp Freddy, DJ Hurricane and John Taylor (Duran Duran).

 Grean was formerly the lead guitarist and touring musical director for Scott Weiland and the Wildabouts.

 Some of Grean's television appearances with Weiland include: VH1 Legends, Last Call with Carson Daly, The Tonight Show with Jay Leno, and the Grammy awards
with Velvet Revolver as keyboardist, performing "All Across the Universe" with Stevie Wonder, Norah Jones, Bono, Steven Tyler, Brian Wilson, and Alison Krauss.
</item 4>

<item 5>
Lazzo (musician)
Lazzo is a producer, engineer, and songwriter specializing in the rock, electronic, dance, and pop genres.

 In late 2013, he released his Wammy (Washington Area Music Association / WAMA) award-winning dance/dubstep remix album with Rites of Ash, Kept Me Up All Night.

 In Oct. 2014, Rites of Ash released their new full-length album titled "Kill For Love".

 He has had his music featured on numerous MTV shows, FUSE, MTV2, and MTVu, as well as shared the stage with many national artists.
</item 5>

<item 6>
Zach Bogosian
In his second season, he tied a team record for goals by a defenseman.

 Bogosian first played in an international tournament when he joined the American national team at the 2009 IIHF World Championship.
</item 6>

<item 7>
Greg Davis (musician)
Greg Davis is an American electronic musician who has recorded albums drawing from a wide variety of sources, including guitar, field recording, various
world/ethnic/traditional instruments, percussion, and voice, all processed through digital manipulation.

 After years of experience in hip-hop groups, jazz combos, free improvisation, and experimental composition, Davis moved toward computer-based music in 1997.

 At Depaul University in Chicago, he studied classical and jazz guitar alongside composition and jazz studies.

 In 1997, he started his own label, Autumn Records, to release his own music and the music of others.

 Davis moved to Boston in 1999 to pursue a master's degree in composition (which he received in June 2001) while independently immersing himself in the study of
electronic music.

 Since 2001, he has played shows all around the world and released records on a number of different labels.

 He has collaborated with Keith Fullerton Whitman, Ben Vida, Sebastien Roux, Chris Weisman, Steven Hess, Jeph Jerman, Akron/Family, Toby Aronson (as
Harmonizer), and Zach Wallace (as Sun Circle).
</item 7>

<item 8>
Daryl Palumbo
Daryl Palumbo (born February 10, 1979) is an American musician, originally from Elmira, New York.

 He is the frontman of the bands Glassjaw, Head Automatica and Color Film.

 As a youth he was a member of the Long Island straight edge band "XbustedX."

 In 1993 he met guitarist Justin Beck, who would become his good friend.

 Together they formed Glassjaw.
</item 8>

<item 9>
Daron Malakian
Daron Vartan Malakian (Armenian: Տարօն/Տարոն Վարդան Մալաքեան, born July 18, 1975) is an Armenian–American singer-songwriter, multi-instrumentalist, and record
producer.

 He is best known as the guitarist, songwriter and second vocalist of the rock band System of a Down and as the lead vocalist, lead guitarist and songwriter of
the band Scars on Broadway.

 Like the rest of the Hollywood-based band System of a Down, he is of Armenian ancestry, but he is the only member to have been born in the United States.

 Daron Malakian is known for his distinctive playing and is ranked 40th in Loudwire's list of "Top 50 Hard Rock + Metal Guitarists Of All Time" and #11 in
MusicRadar's poll, "The 20 Greatest Metal Guitarists Ever".

 He is placed 30th in Guitar World's List of The 100 Greatest Heavy Metal Guitarists of All Time.
</item 9>

<item 10>
Zach Bogosian
Zachary M. Bogosian (Armenian: Զաքարի Մ.

 Պողոսյան , born July 15, 1990) is an American professional ice hockey defenseman currently playing for the Buffalo Sabres.

 Bogosian attended Cushing Academy in Massachusetts before he joined the Peterborough Petes of the Ontario Hockey League (OHL) in 2006.

 He played two seasons in the OHL, and was nominated for the Red Tilson Trophy as the league's most outstanding player in his second season.

 Bogosian was regarded as a complete, physical defenseman who could contribute on both offense and defense; he was rated as one of the top players heading into
the 2008 NHL Entry Draft, where the Atlanta Thrashers selected him third overall.

 He signed a contract with the team a few weeks after the draft and began the season with the Thrashers, though he missed several weeks of his rookie year due
to injury.

 In his second season, he tied a team record for goals by a defenseman.
</item 10>
=== END ITEM id=2e7ce0f02d2f6f1a ===

=== ITEM id=8947a26c25a5d2cd ===
Gold sentences and facts:
Sentence 0:  Her only novel, "The Bone People", won the Booker Prize in 1985.
  Fact 0.0: Her only novel is 'The Bone People'.
  Fact 0.1: 'The Bone People' won the Booker Prize.
  Fact 0.2: 'The Bone People' won the Booker Prize in 1985.
Sentence 1:  His novel "Ender's Game" (1985) and its sequel "Speaker for the Dead" (1986) both won Hugo and Nebula Awards, making Card the only author to win both science fiction's top U.S. prizes in consecutive years.
  Fact 1.0: Card's novel 'Ender's Game' was published in 1985.
  Fact 1.1: 'Speaker for the Dead' is the sequel to 'Ender's Game'.
  Fact 1.2: 'Speaker for the Dead' was published in 1986.
  Fact 1.3: 'Ender's Game' won the Hugo Award.
  Fact 1.4: 'Ender's Game' won the Nebula Award.
  Fact 1.5: 'Speaker for the Dead' won the Hugo Award.
  Fact 1.6: 'Speaker for the Dead' won the Nebula Award.
  Fact 1.7: Card is the only author to win both science fiction's top U.S. prizes in consecutive years.

Retrieved items:
<item 1>
Orson Scott Card
Orson Scott Card (born August 24, 1951) is an American novelist, critic, public speaker, essayist, and columnist.

 He writes in several genres but is known best for science fiction.

 His novel "Ender's Game" (1985) and its sequel "Speaker for the Dead" (1986) both won Hugo and Nebula Awards, making Card the only author to win both science
fiction's top U.S. prizes in consecutive years.

 A feature film adaptation of "Ender's Game", which Card co-produced, was released in late October 2013 in Europe and on November 1, 2013, in North America.
</item 1>

<item 2>
The Bone People
The Bone People (styled by the writer and in some editions as the bone people) is a Booker Prize-winning 1984 novel by New Zealand writer Keri Hulme.
</item 2>

<item 3>
A War of Gifts: An Ender Story
A War of Gifts: An Ender Story (2007) is a science fiction novel by Orson Scott Card.

 This book is set in Card's Ender's Game series and takes place during Ender Wiggin's time at Battle School as described in Card's novels "Ender's Game" and
"Ender's Shadow".
</item 3>

<item 4>
Ender Wiggin
Andrew "Ender" Wiggin is a fictional character from Orson Scott Card's 1985 science fiction novel "Ender's Game" and its sequels ("Speaker for the Dead",
"Xenocide", "Children of the Mind", "Ender in Exile"), as well as in the first part of the spin-off series, "Ender's Shadow".

 The book series itself is an expansion, with some changes to detail, of Card's 1977 short story "Ender's Game."
</item 4>

<item 5>
Ender in Exile
Ender in Exile is a science fiction novel by Orson Scott Card, part of the "Ender's Game" series, published on November 11, 2008.

 It takes place between the two award-winning novels: "Ender's Game" and "Speaker for the Dead".

 It could also be considered a parallel novel to the first three sequels in the Shadow Saga, since the entirety of this "trilogy" takes place in the span of
"Ender in Exile".

 The novel concludes a dangling story line of the Shadow Saga, while it makes several references to events that take place during the Shadow Saga.

 From yet another perspective, the novel expands (or "replaces") the last chapter of the original novel "Ender's Game".

 On the one hand, it fills the gap right before the last chapter, and on the other hand, it fills the gap between the last chapter and the original (first)
sequel (both named "Speaker for the Dead").
</item 5>

<item 6>
Orson Scott Card bibliography
This is a list of the works of Orson Scott Card.

 This list does not include criticisms, reviews, or related material written by Card.

 Orson Scott Card is the author of The Ender saga and Homecoming Saga among many other works.
</item 6>

<item 7>
Ender in Exile
"Ender in Exile" begins one year after Ender has won the bugger war, and begins with the short story "Ender's Homecoming" from Card's webzine Intergalactic
Medicine Show.

 Other short stories that were published elsewhere are included as chapters of the novel.
</item 7>

<item 8>
Ender's Game (short story)
"Ender's Game" is a story by Orson Scott Card.

 It first appeared in the August 1977 issue of "Analog" magazine and was later expanded into the novel "Ender's Game".

 Although the foundation of the Ender's Game series, the short story is not properly part of the "Ender's Game" universe, as there are many discrepancies in
continuity.
</item 8>

<item 9>
List of Ender's Game series short stories
There are various sources for short stories set in the Ender's Game series.

 One is the short story collection "First Meetings" by Orson Scott Card.

 This collection contains the original novelette "Ender's Game" plus three other stories.

 Another source is Card’s webzine "InterGalactic Medicine Show".

 The first four stories from Card's webzine: "Mazer in Prison," "Pretty Boy," "Cheater," and "A Young Man with Prospects," also appear in the paperback
anthology "Orson Scott Card's InterGalactic Medicine Show".

 Reprints of short stories in the Ender's Game series can be found in other science fiction anthologies.
</item 9>

<item 10>
Shadow of the Hegemon
Shadow of the Hegemon (2001) is the second novel in the "Ender's Shadow" series (often called the Bean Quartet) by Orson Scott Card.

 It is also the sixth novel in the "Ender's Game" series.

 It is told mostly from the point of view of Bean, a largely peripheral character in the original novel "Ender's Game" but the central protagonist of the
parallel narrative "Ender's Shadow."

 "Shadow of the Hegemon" was nominated for a Locus Award in 2002.
</item 10>
=== END ITEM id=8947a26c25a5d2cd ===

=== ITEM id=3bf31d13d4b15557 ===
Gold sentences and facts:
Sentence 0:  He finished second to Paul Hornung in voting for the Heisman Trophy in 1956.
  Fact 0.0: The person finished second in voting for the Heisman Trophy in 1956.
  Fact 0.1: Paul Hornung finished first in that Heisman Trophy voting in 1956.
Sentence 1: Paul Vernon Hornung (born December 23, 1935), nicknamed "The Golden Boy", is a former professional American football player, a Hall of Fame running back for the Green Bay Packers of the National Football League (NFL) from 1957 to 1966, winning four NFL titles and the first Super Bowl.
  Fact 1.0: Paul Vernon Hornung was born on December 23, 1935.
  Fact 1.1: Paul Vernon Hornung is nicknamed "The Golden Boy".
  Fact 1.2: Paul Vernon Hornung is a former professional American football player.
  Fact 1.3: Paul Vernon Hornung is a Hall of Fame running back.
  Fact 1.4: Paul Vernon Hornung played for the Green Bay Packers of the National Football League (NFL).
  Fact 1.5: Paul Vernon Hornung played for the Green Bay Packers from 1957 to 1966.
  Fact 1.6: Paul Vernon Hornung won four NFL titles.
  Fact 1.7: Paul Vernon Hornung won the first Super Bowl.

Retrieved items:
<item 1>
Johnny Majors
John Terrill Majors (born May 21, 1935) is a former American football player and coach.

 A standout halfback at the University of Tennessee, he was an All-American in 1956 and a two-time winner of the Southeastern Conference Most Valuable Player
award, in 1955 and 1956.

 He finished second to Paul Hornung in voting for the Heisman Trophy in 1956.

 Majors served as the head football coach at Iowa State University (1968–1972), the University of Pittsburgh (1973–1976, 1993–1996), and Tennessee (1977–1992),
compiling a career college football record of 185–137–10.

 His 1976 Pittsburgh squad won a national championship after capping a 12–0 season with a victory in the Sugar Bowl.

 Majors was inducted into the College Football Hall of Fame as a player in 1987.
</item 1>

<item 2>
1976 NCAA Division I football season
The 1976 NCAA Division I football season ended with a championship for the Panthers of the University of Pittsburgh.

 Led by head coach Johnny Majors (voted the AFCA Coach of the Year), the Pitt Panthers brought a college football championship to the home of the defending pro
football champions, the Steelers.

 Pitt also had the Heisman Trophy winner, Tony Dorsett; the Panthers had been ranked ninth in the preseason AP poll.
</item 2>

<item 3>
2012 Texas A&amp;M vs. Alabama football game
The 2012 Texas A&M vs. Alabama football game was a college football game between the Texas A&M Aggies and Alabama Crimson Tide at Bryant–Denny Stadium in
Tuscaloosa, Alabama.

 This was the game where Texas A&M freshman quarterback and eventual Heisman Trophy winner Johnny Manziel completed 24 of 31 passes, had 253 passing yards, and
two passing touchdowns with 92 rushing yards to help No. 15 Texas A&M upset No. 1 Alabama 29–24, which led him to being the first freshman to win the Heisman
Trophy.
</item 3>

<item 4>
Heisman Trophy
The Heisman Memorial Trophy (usually known colloquially as the Heisman Trophy or The Heisman), is awarded annually to the most outstanding player in college
football in the United States whose performance best exhibits the pursuit of excellence with integrity.

 Winners epitomize great ability combined with diligence, perseverance, and hard work.

 It is presented by the Heisman Trophy Trust in early December before the postseason bowl games.
</item 4>

<item 5>
Graph query row:
person: Paul Vernon Hornung
birth_date: December 23, 1935
award_relationship: AWARDED
award: Heisman Trophy
award_year: None
Johnny_Majors: John Terrill Majors
Johnny_Majors_award_relationship: None
</item 5>

<item 6>
Johnny Bright
Johnny D. Bright (June 11, 1930 – December 14, 1983) was a professional Canadian football player in the Canadian Football League.

 He played college football at Drake University.

 He is a member of the Canadian Football Hall of Fame, the National Football Foundation's College Football Hall of Fame, the Missouri Valley Conference Hall of
Fame, the Edmonton Eskimos Wall of Honour, the Alberta Sports Hall of Fame, and the "Des Moines Register's" Iowa Sports Hall of Fame.
</item 6>

<item 7>
Frank Eliscu
Frank Eliscu (1912–1996) was an American sculptor and art teacher who designed and created the Heisman Memorial Football Trophy in 1935 when he was only 20
years old.

 The first Heisman Trophy, a strong young bull of a football player cast in bronze, was presented to a college football player in 1935, and is considered one of
the greatest honors a college athlete can receive.

 Over the years the Heisman Committee has paid tribute to Eliscu and his creation several times, and in 1985, the Heisman Committee invited him to speak at the
50th Anniversary Heisman ceremonies.
</item 7>

<item 8>
Heisman curse
Although many Heisman winners have not enjoyed success at the professional level, including players like Matt Leinart, Andre Ware, Jason White, Rashaan Salaam,
Eric Crouch, Ty Detmer, Troy Smith and Gino Torretta, proponents of the "curse" rarely cite highly successful players such as Barry Sanders, Charles Woodson,
Eddie George, Tim Brown, Bo Jackson, Marcus Allen, Earl Campbell, and Tony Dorsett among the notables.
</item 8>

<item 9>
1997 Tennessee Volunteers football team
However, the Vols' defense could not stop Nebraska's rushing attack, giving up more than 400 yards on the ground in a 42–17 loss.

 As a senior, Manning won numerous awards.

 He was a consensus first-team All-American and won the Maxwell Award, the Davey O'Brien Award, the Johnny Unitas Award, and the Best College Football Player
ESPY Award, among others.

 However, he did not win the Heisman Trophy, finishing runner-up to Charles Woodson, a CB from Michigan, and the only defensive player ever to win the Heisman
Trophy.
</item 9>

<item 10>
Paul Hornung
Paul Vernon Hornung (born December 23, 1935), nicknamed "The Golden Boy", is a former professional American football player, a Hall of Fame running back for the
Green Bay Packers of the National Football League (NFL) from 1957 to 1966, winning four NFL titles and the first Super Bowl.

 He is the first in pro football history to win the Heisman Trophy, be selected as the first overall selection in the NFL Draft, win the NFL most valuable
player award, and be inducted into both the professional and college football halls of fame.
</item 10>
=== END ITEM id=3bf31d13d4b15557 ===

=== ITEM id=d8c1f1d3e12df7d9 ===
Gold sentences and facts:
Sentence 0: Paul Clyne was the District Attorney of Albany County, New York from January 2001 through December 2004.
  Fact 0.0: Paul Clyne was the District Attorney of Albany County, New York.
  Fact 0.1: Paul Clyne held this office from January 2001.
  Fact 0.2: Paul Clyne held this office through December 2004.
Sentence 1: P. David Soares (born October 26, 1969, Brava, Cape Verde) is the Albany County, N.Y. District Attorney.
  Fact 1.0: P. David Soares was born on October 26, 1969.
  Fact 1.1: P. David Soares was born in Brava, Cape Verde.
  Fact 1.2: P. David Soares is the District Attorney of Albany County, New York.

Retrieved items:
<item 1>
Paul Clyne
Paul Clyne was the District Attorney of Albany County, New York from January 2001 through December 2004.

 A graduate of Albany Law School, he spent about 14 years as an assistant district attorney, before he was tapped by local politicians to replace the retiring
District Attorney, Sol Greenberg.

 He was defeated for re-election by David Soares, first in the Democratic Party primary election in September 2004, and then in the general election in November
2004, in which he ran on an independent line.

 After a stint teaching at the New York Prosecutors Institute, he went into private practice as a criminal defense attorney in 2007, with an office in Albany,
New York.
</item 1>

<item 2>
Two Rivers Press
Two Rivers Press is an independent publishing house, based in the English town of Reading.

 Two Rivers Press was founded in 1994 by Peter Hay (1951–2003), a local artist.

 Its name reflects his enthusiasm for the town and its two rivers, the Kennet and the Thames, and its intention to explore "the place where art and history
meet".

 The name also gives a clue to the origins of the company in the, ultimately successful, opposition to Reading's proposed "Cross-Town Route", a road scheme that
would have seriously impacted the point at which the two rivers meet.
</item 2>

<item 3>
Câmara de Lobos
Câmara de Lobos (] ; literally, Portuguese: "chamber of the sealions" ) is a municipality, parish and town in the south-central coast of the island of Madeira.

 Technically a suburb of the much larger capital city of Funchal, it is one of the larger population centres and an extension of the Funchal economy.
</item 3>

<item 4>
Skip Bolen
Skip Bolen is a Southern photographer of musicians, architecture, lifestyle and the culture of New Orleans.

 Born in Lafayette, Louisiana, he moved to New Orleans where he began his publishing career as a designer and art director.

 After moving to New York City, he began working at "House & Garden," renamed "HG," as Senior Designer in January 1988 with Anna Wintour and Alexander Liberman
at Condé Nast Publications.

 Spending evenings in jazz clubs, he began photographing jazz musicians in New York City and often when he regularly returned to New Orleans.

 After three years at Condé Nast Publications in New York City, he returned to New Orleans to pursue his jazz photography full-time.

 In 1998, he moved to Los Angeles where he became art director of House of Blues for seven years while photographing at night and weekends.

 He continued photographing jazz musicians and had his first major solo exhibition at the Jazz Bakery in Los Angeles on August 9, 2002.

 On July 4, 2006, he returned to New Orleans to pursue photography full-time documenting the recovery and rebuilding of New Orleans since Hurricane Katrina,
documenting the jazz scene, night-time photography and other photographic projects.
</item 4>

<item 5>
Paul Soniat
Paul Soniat is the Director of the New Orleans City Park Botanical Gardens.

 He was born at the old Touro Hospital in New Orleans.

 His family has been in New Orleans since 1727 and he grew up and later lived on the street bearing his family name.

 Paul is a self-taught piano player, and has released two CDs, absorbing the sights, sounds, and flavor of New Orleans.

 His first CD, titled "Born in New Orleans", was released in April 2005.

 His second CD was released after Hurricane Katrina hit the Gulf coast and the New Orleans' levees failed.

 This second CD is appropriately titled "Below the Water Line".
</item 5>

<item 6>
Swansea City Opera
The Swansea City Opera is a touring opera company founded in Swansea, Wales in May 2004.

 The company incorporates elements of Opera Box Limited, a touring opera company founded in 1989.

 Since its launch the opera has toured to 91 venues across the United Kingdom.

 The company places particular emphasis on touring within Wales, especially within Objective 1 and Communities First areas where performances are given at
subsidised cost.
</item 6>

<item 7>
David Soares
P. David Soares (born October 26, 1969, Brava, Cape Verde) is the Albany County, N.Y. District Attorney.

 He is a Democrat.
</item 7>

<item 8>
Richard Pleasance
Richard Arnold Pleasance is an Australian rock musician and producer.

 He was a founding member of Boom Crash Opera on guitar, bass guitar, vocals and as a songwriter in 1985; they released three albums before Pleasance left in
1992.

 Their hit Australian Recording Industry Association (ARIA) singles, "Great Wall" (No. 5, 1986) and "Onion Skin" (No. 11, 1989) were co-written by Pleasance,
who also co-produced their second album, "These Here Are Crazy Times" (No. 10, 1989).

 His debut solo release, "Galleon" received four nominations at the ARIA Music Awards for 1992.

 Pleasance composed the theme music for Australian television series, "SeaChange" (1998–2001), for 2006 feature film "Kenny" and more recently he composed the
theme music for the prison drama series "Wentworth".

 Pleasance is married to Michelle and, as from May 2009, he was living in Hepburn Springs, Victoria where he has a recording studio.
</item 8>

<item 9>
City Cruises
City Cruises is a public limited company that operates scheduled public sightseeing cruises on the River Thames in London, serving Westminster, London Eye,
Tower and Greenwich piers every day throughout the year.

 It operates the services mainly using purpose-built wheelchair-accessible 'RiverLiner' vessels, which have a large open top deck and enclosed lower saloon with
panoramic windows and bars, backed up by some ex-Paris single-deck conservatory-style sightseeing boats.

 The company also owns a number of other boats with which it provides a variety of private charter and waterage services on the River Thames.

 City Cruises owns the London Showboat brand, under which it operates an evening dinner, cabaret and dance cruise from Westminster Pier throughout the year, and
also owns and operates the Restaurant Ship Hispaniola, which is permanently moored alongside Victoria Embankment close to Embankment Underground station in
central London.
</item 9>

<item 10>
Dave Hemingway
Dave Hemingway (born David Robert Hemingway, 20 September 1960) is an English musician and songwriter, and was a vocalist for the Hull-based band The Beautiful
South until they disbanded in 2007.

 Previously he had been a member of The Housemartins.
</item 10>
=== END ITEM id=d8c1f1d3e12df7d9 ===

=== ITEM id=e7c1be501c146884 ===
Gold sentences and facts:
Sentence 0:  He received an excellent education in both Arabic and Greek, having studied Arabic grammar with the famous poet Abul ʿAla Al-Maʿarri.
  Fact 0.0: He received an excellent education in Arabic.
  Fact 0.1: He received an excellent education in Greek.
  Fact 0.2: He studied Arabic grammar with the famous poet Abul ʿAla Al-Maʿarri.
Sentence 1: Abul ʿAla Al-Maʿarri (Arabic أبو العلاء المعري "Abū al-ʿAlāʾ al-Maʿarrī ", full name أبو العلاء أحمد بن عبد الله بن سليمان التنوخي المعري "Abū al-ʿAlāʾ Aḥmad ibn ʿAbd Allāh ibn Sulaimān al-Tanūẖī al-Maʿarrī "; December 973 – May 1057) was a blind Arab philosopher, poet, and writer.
  Fact 1.0: Abul ʿAla Al-Maʿarri's Arabic name is أبو العلاء المعري (Abū al-ʿAlāʾ al-Maʿarrī).
  Fact 1.1: Abul ʿAla Al-Maʿarri's full name is أبو العلاء أحمد بن عبد الله بن سليمان التنوخي المعري (Abū al-ʿAlāʾ Aḥmad ibn ʿAbd Allāh ibn Sulaimān al-Tanūẖī al-Maʿarrī).
  Fact 1.2: Abul ʿAla Al-Maʿarri was born in December 973.
  Fact 1.3: Abul ʿAla Al-Maʿarri died in May 1057.
  Fact 1.4: Abul ʿAla Al-Maʿarri was blind.
  Fact 1.5: Abul ʿAla Al-Maʿarri was an Arab philosopher.
  Fact 1.6: Abul ʿAla Al-Maʿarri was an Arab poet.
  Fact 1.7: Abul ʿAla Al-Maʿarri was an Arab writer.

Retrieved items:
<item 1>
Al-Mutanabbi
Abu at-Tayyib Ahmad bin Al-Husayn al-Mutanabbi al-Kindi (Arabic: أبو الطيب أحمد بن الحسين المتنبّي الكندي‎ , "Abū ṭ-Ṭayyib ʾAḥmad bin al-Ḥusayn al-Muṫanabbī
al-Kindī " ‎ ) (915 – 23 September 965 CE) was an Arab poet.

 He is considered as one of the greatest poets in the Arabic language and is the most prominent and most influential poet in the Arab world and much of his work
has been translated into over 20 languages worldwide.

 Much of his poetry revolves around praising the kings he visited during his lifetime.

 Some consider his 326 poems to be a great representation of his life story.

 He started writing poetry when he was nine years old.

 He is well known for his sharp intelligence and wittiness.

 Al-Mutanabbi had a great pride in himself through his poetry.
</item 1>

<item 2>
Umar Ibn Abi Rabi'ah
'Umar ibn Abi Rabi'ah al-Makhzumi (November 644, Mecca – 712/719, Mecca, full name: Abū ’l-Khattāb Omar Ibn Abd Allah Ibn Abi Rabia Ibn al-Moghaira Ibn Abd
Allah Ibn Omar Ibn Makhzūm Ibn Yakaza Ibn Murra al-Makhzūmi) was an Arabic poet.

 He was born into a wealthy family of the Quraish tribe of Mecca, his father being Abd Allah and his mother Asmā bint Mukharriba.

 He was characterised by the biographer Ibn Khallikan as 'the best poet ever produced by the tribe of Koraish'.
</item 2>

<item 3>
Shaykh Muhammad ibn Thaleb
Muhammad ibn Thalib ibn Abd Allah ibn Ni`mat Allah ibn Sadr ad-Din ibn Shaykh Baha' ad-Din ash-Shirazi (Arabic: محمد ابن طالب عبدالله ابن نعمات الله ابن صدر
الدين ابن شيخ بهاءالدين الشيرازي‎ ‎ ) was a 15th-century Persian physician from Shiraz, Iran.
</item 3>

<item 4>
Abd Allah ibn al-Fadl
Little is known of his life, apart from what can be gleaned from manuscripts of his texts.

 He was a deacon and the grandson of a bishop.

 He received an excellent education in both Arabic and Greek, having studied Arabic grammar with the famous poet Abul ʿAla Al-Maʿarri.

 Several of his works and translations were commissioned by notables from Antioch and neighbouring cities in Muslim territory.
</item 4>

<item 5>
Al-Mutanabbi
Al-Mutanabbi had a great pride in himself through his poetry.

 Among the topics he discussed were courage, the philosophy of life, and the description of battles.

 Many of his poems were and still are widely spread in today's Arab world and are considered to be proverbial.
</item 5>

<item 6>
Ibn al-Khatib
Lisan ad-Din ibn al-Khatib (Born 16 November 1313, Loja– died 1374, Fes, Morocco) (Full name Muhammad ibn Abd Allah ibn Said ibn Ali ibn Ahmad al-Salmani) was a
Moroccan Arab polymath poet, writer, historian, philosopher, physician and politician from Emirate of Granada.

 Some of his poems decorate the walls of the Alhambra in Granada.
</item 6>

<item 7>
Abd Allah ibn Abbas
Abd Allah ibn Abbas (Arabic: عبد الله ابن عباس‎ ‎ ) or ′Abd Allah ibn al-′Abbas otherwise called (Ibn Abbas; Al-Habr; Al-Bahr; The Doctor; The Sea) was born c.
619 CE.

 He was the son of Al-‘Abbas ibn ‘Abd al-Muttalib, an uncle of the Islamic prophet Muhammad, and a nephew of the Maymunah bint al-Harith, who later became
Muhammad's wife.

 He was one of Muhammad's cousins and one of the early Qur'an scholars.
</item 7>

<item 8>
Abdal
Abdāl ("lit": substitutes, but which can also mean "generous" ["karīm"] and "noble" ["sharīf"]) is a term used in Islamic metaphysics and Islamic mysticism,
both Sunni and Shiite, to refer to a particularly important group of God's saints.

 In the tradition of Sunni Islam in particular, the concept attained an especially important position in the writings of the Sunni mystics and theologians,
whence it appears in the works of Sunni authorities as diverse as Abu Talib al-Makki (d. 956), Ali Hujwiri (d. 1072), Ibn Asakir (d. 1076), Khwaja Abdullah
Ansari (d. 1088), and Ibn Khaldun (d. 1406).
</item 8>

<item 9>
Al-Maʿarri
Abul ʿAla Al-Maʿarri (Arabic أبو العلاء المعري "Abū al-ʿAlāʾ al-Maʿarrī ", full name أبو العلاء أحمد بن عبد الله بن سليمان التنوخي المعري "Abū al-ʿAlāʾ Aḥmad
ibn ʿAbd Allāh ibn Sulaimān al-Tanūẖī al-Maʿarrī "; December 973 – May 1057) was a blind Arab philosopher, poet, and writer.

 Al-Maʿarri held and expressed an irreligious world view which was met with controversy, but in spite of it, he is regarded as one of the greatest classical
Arabic poets.
</item 9>

<item 10>
Da'ud Abu al-Fadl
Da'ud Abu al-Fadl (1161–1242) was a Karaite Jewish physician who lived in the Ayyubid Egypt in the twelfth century CE.

 He born in Cairo in 1161 and died there about 1242.

 Having studied medicine under the Jewish physician Hibat Allah ibn Jami, and under Abu al-Fafa'il ibn Naqid, he became the court physician of the sultan
al-Malik al-'Adil Abu Bakr ibn Ayyub, the brother and successor of Saladin.

 He was also chief professor at the al-Nasiri Hospital at Cairo, where he had a great many pupils, among them being the historian ibn Abi Usaibiyyah.

 The latter declared that Abu al-Fadl was the most skillful physician of the time and that his success in curing the sick was miraculous.

 Abu al-Fadl was the author of an Arabic pharmacopoeia in twelve chapters, entitled "Aḳrabadhin", and treating chiefly of antidotes.
</item 10>
=== END ITEM id=e7c1be501c146884 ===

=== ITEM id=98c3a9c54773de1d ===
Gold sentences and facts:
Sentence 0: Tritonia (flame freesia) is a genus of flowering plants in the iris family first described as a genus in 1802.
  Fact 0.0: Tritonia, also called flame freesia, is a genus of flowering plants.
  Fact 0.1: Tritonia belongs to the iris family.
  Fact 0.2: Tritonia was first described as a genus in 1802.
Sentence 1: Phyteuma is a genus of flowering plants in the family Campanulaceae, native to Europe and Morocco.
  Fact 1.0: Phyteuma is a genus of flowering plants.
  Fact 1.1: Phyteuma is in the family Campanulaceae.
  Fact 1.2: Phyteuma is native to Europe.
  Fact 1.3: Phyteuma is native to Morocco.

Retrieved items:
<item 1>
Thonningia
Thonningia is a monotypic genus of flowering plants in the family Balanophoraceae containing the single species Thonningia sanguinea.

 It is distributed throughout much of southern and western Africa, particularly the tropical regions.

 Common names for the plant include ground pineapple.

 A familiar plant to humans, it has an extremely long list of common names in many African languages.

 Many names are inspired by the resemblance of the plant's inflorescence to a pineapple or palm tree.

 Some of the names can be translated as pineapple of the bush (from Anyi), duiker's kolanut (from Igala), and crown of the ground (from Yoruba).
</item 1>

<item 2>
Phyteuma
Phyteuma is a genus of flowering plants in the family Campanulaceae, native to Europe and Morocco.
</item 2>

<item 3>
Alismataceae
The water-plantains (Alismataceae) are a family of flowering plants, comprising 11 genera and between 85 and 95 species.

 The family has a cosmopolitan distribution, with the greatest number of species in temperate regions of the Northern Hemisphere.

 Most of the species are herbaceous aquatic plants growing in marshes and ponds.
</item 3>

<item 4>
Zygotritonia
Zygotritonia is a genus of herbaceous, perennial and bulbous plants in the Iris family (Iridaceae).

 It contains four species distributed in Tropical Africa.

 The genus name is derived from the word zygomorphic (meaning bilabiate), and the apparent resemblance to some species in the genus "Tritonia".
</item 4>

<item 5>
Tritonia (plant)
Tritonia (flame freesia) is a genus of flowering plants in the iris family first described as a genus in 1802.

 They are naturally distributed across southern Africa, with a high concentration of species in Cape Province of western South Africa.

 The genus is closely related to the genus "Ixia".
</item 5>

<item 6>
Conophytum
Conophytum is a genus of South African and Namibian succulent plants that belong to the Aizoaceae family.

 The name is derived from the Latin “conus” (cone) and Greek “phytum” (plant).

 The plants are also known as knopies (buttons), waterblasies (water blisters), sphaeroids, conos, cone plants, dumplings, or button plants.
</item 6>

<item 7>
Tricyrtis
Tricyrtis is a genus of Asian flowering plants in the lily family, with approximately 20 known species.

 The species are commonly known in English as toad lilies.

 The genus has a native range from the Himalayas to eastern Asia, including China, Japan, Philippines and Taiwan, and a few species are cultivated for their
ornamental qualities in other parts of the world.
</item 7>

<item 8>
Tritoniopsis
Tritoniopsis is a genus of flowering plants in the family Iridaceae, first described as a genus in 1926.

 The entire genus is endemic to Cape Province in South Africa.

 The genus name refers to the African genus "Tritonia" and is combined with the Greek word "opsis", meaning "look-alike".
</item 8>

<item 9>
Vascular plant
Vascular plants (from Latin "vasculum": duct), also known as tracheophytes (from the equivalent Greek term "trachea") and also higher plants, form a large group
of plants (c. 308,312 accepted known species ) that are defined as those land plants that have lignified tissues (the xylem) for conducting water and minerals
throughout the plant.

 They also have a specialized non-lignified tissue (the phloem) to conduct products of photosynthesis.

 Vascular plants include the clubmosses, horsetails, ferns, gymnosperms (including conifers) and angiosperms (flowering plants).

 Scientific names for the group include Tracheophyta and Tracheobionta.
</item 9>

<item 10>
Cyrtanthus
Cyrtanthus is a genus of perennial, herbaceous and bulbous plants in the family Amaryllidaceae, subfamily Amaryllidoideae.
</item 10>
=== END ITEM id=98c3a9c54773de1d ===

=== ITEM id=4cfa7a1a8f78e2c9 ===
Gold sentences and facts:
Sentence 0:  He is most well known for providing the singing voice of Tony (played onscreen by Richard Beymer) in the 1961 film musical "West Side Story".
  Fact 0.0: This person is most well known for providing the singing voice of Tony in the 1961 film musical "West Side Story".
  Fact 0.1: Richard Beymer played Tony onscreen in the 1961 film musical "West Side Story".
Sentence 1:  The film was nominated for 11 Academy Awards and won 10, including Best Picture (as well as a special award for Robbins), becoming the record holder for the most wins for a movie musical.
  Fact 1.0: West Side Story was nominated for 11 Academy Awards.
  Fact 1.1: West Side Story won 10 Academy Awards.
  Fact 1.2: West Side Story won Best Picture.
  Fact 1.3: West Side Story won a special award for Robbins.
  Fact 1.4: West Side Story became the record holder for the most wins for a movie musical.

Retrieved items:
<item 1>
Academy Honorary Award
The Academy Honorary Award is often awarded in preference to those with noted achievements in motion pictures who have nevertheless never won an Academy Award.

 Thus, many of its recipients are Classic Hollywood stars, such as Lillian Gish, Barbara Stanwyck, Kirk Douglas, and Lauren Bacall.
</item 1>

<item 2>
F. Keogh Gleason
Francis Keogh Gleason (April 14, 1906 – December 18, 1982) was a resident set decorator at MGM studios for over 40 years.

 In that time he won 4 Academy Awards (for "An American in Paris" in 1951, "The Bad and the Beautiful" in 1952, "Somebody Up There Likes Me" in 1956 and "Gigi"
in 1958) and was nominated an additional 3 times.

 Gleason graduated from the Minnesota School of Fine Arts (now known as the Minneapolis College of Art and Design) in 1927.
</item 2>

<item 3>
Romeo and Juliet (1968 film soundtrack)
The soundtrack for the 1968 film "Romeo and Juliet" was composed and conducted by Nino Rota.

 It was originally released as a vinyl record, containing nine entries, most notably the song "What Is a Youth", composed by Nino Rota, written by Eugene Walter
and performed by Glen Weston.

 The music score won a Silver Ribbon award of the Italian National Syndicate of Film Journalists in 1968 and was nominated for two other awards (BAFTA Award for
Best Film Music in 1968 and Golden Globe Award for Best Original Score in 1969).
</item 3>

<item 4>
Triple Crown of Acting
The Triple Crown of Acting is a term used in the American entertainment industry to describe actors who have won a competitive Academy Award, Emmy Award, and
Tony Award in the acting categories.

 23 people have achieved the triple crown of acting (14 women, 9 men).

 Helen Hayes' Emmy Award win on February 5, 1953, made her the first person to achieve the triple crown.

 Thomas Mitchell became the first man to achieve the triple crown with his Tony Award win later the same year on March 29, 1953.

 Hayes and Rita Moreno are the only triple crown winners in competitive acting categories who have also won a Grammy Award to complete the EGOT.
</item 4>

<item 5>
Jimmy Bryant (singer)
James Howard Bryant (born June 2, 1929) is a singer, arranger and composer.

 He is most well known for providing the singing voice of Tony (played onscreen by Richard Beymer) in the 1961 film musical "West Side Story".

 While he received no screen credit, he states that Beymer was "a nice guy, and every time he did an interview he would mention my name."

 He also sang for James Fox in the 1967 film musical "Thoroughly Modern Millie", and sang in "The Telephone Hour" number in "Bye Bye Birdie".

 He also sang in the group that performed the theme song of the TV series "Batman".
</item 5>

<item 6>
The Godfather (soundtrack)
The Godfather is the soundtrack from the film of the same name, released in 1972 by Paramount Records, and in 1991 on compact disc by MCA.

 Unless noted, the cues were composed by Nino Rota and conducted by Carlo Savina (who was credited on the LP, but not the CD).

 The song "I Have but One Heart" is sung by Al Martino, who performed it in the film as character Johnny Fontane.
</item 6>

<item 7>
Dean Martin Sings
Dean Martin Sings is the first studio album by Italian-American singer Dean Martin, released in 1953.

 It is the first long-play 10-inch album recorded by Martin for Capitol Records during two sessions recorded on the evening of November 20, 1952.

 The first session was recorded between 5 and 8 PM and it produced five songs featuring string arrangements.

 "There's My Lover" was recorded but not released.

 After a ninety-minute break, Martin was joined by a brass arrangement to record the remaining four songs.

 Seven of the eight songs on this album appeared in the Martin & Lewis film, "The Stooge".

 Two years later, the songs from this 10-inch album would be combined with four newly popular songs recorded between 1951 and 1953 to create a full-length
12-inch album.

 The 2005 Collectors' Choice reissue added four bonus songs recorded between 1949 and 1953 and was released with alternative cover artwork.
</item 7>

<item 8>
Academy Honorary Award
The Academy Honorary Award – instituted in 1948 for the 21st Academy Awards (previously called the Special Award, which was first presented in early 1929) – is
given annually by the Board of Governors of the Academy of Motion Picture Arts and Sciences (AMPAS) to celebrate motion picture achievements that are not
covered by existing Academy Awards, although prior winners of competitive Academy Awards are not excluded from receiving the Honorary Award.

 Unless otherwise specified, Honorary Award recipients receive the same gold Oscar statuettes received by winners of the competitive Academy Awards.

 Unlike the Special Achievement Award instituted in 1972 (and discontinued in 1995), those on whom the Academy confers its Honorary Award do not have to meet
"the Academy's eligibility year and deadline requirements."

 Like the Special Achievement Award, the Special Award and Honorary Award have been used to reward significant achievements of the year that did not fit in
existing categories, subsequently leading the Academy to establish several new categories, and to honor exceptional career achievements, contributions to the
motion picture industry, and service to the Academy.
</item 8>

<item 9>
Geoffrey Rush
Geoffrey Roy Rush {'1': ", '2': ", '3': ", '4': "} (born 6 July 1951) is an Australian actor and film producer.

 Rush is the youngest amongst the few people who have won the "Triple Crown of Acting": the Academy Award, the Primetime Emmy Award, and the Tony Award.

 He has won one Academy Award for acting (from four nominations), three British Academy Film Awards (from five nominations), two Golden Globe Awards and four
Screen Actors Guild Awards. Rush is the founding President of the Australian Academy of Cinema and Television Arts and was named the 2012 Australian of the
Year.

 He is also the first actor to win the Academy Award, BAFTA Award, Critics' Choice Movie Award, Golden Globe Award, and Screen Actors Guild Award for a single
performance in film for his performance in "Shine" (1996).
</item 9>

<item 10>
Tony Hiller
Anthony Toby "Tony" Hiller (born 30 July 1927) is a British songwriter.

 He began his musical career as a member of the song and dance duo The Hiller Brothers, sharing the stage with his brother Irving.

 The Hiller Brothers appeared with many artists of the time including Alma Cogan, Tommy Cooper, Val Doonican, Matt Monro, The Shadows, Bernard Manning, Kathy
Kirby, Roger Whittaker, Rip Taylor, Gene Vincent, Lance Percival, Tessie O'Shea, Frank Ifield, Deep River Boys, The Dallas Boys, Clark Brothers, Paul Melba and
Ray Burns.
</item 10>
=== END ITEM id=4cfa7a1a8f78e2c9 ===
