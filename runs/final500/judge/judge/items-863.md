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

This file has 6 items.

=== ITEM id=85022bfb40766a99 ===
Gold sentences and facts:
Sentence 0: In every High Court , there is a Chief Justice and many other judges whose number is defined by the President of India .
  Fact 0.0: Every High Court has a Chief Justice.
  Fact 0.1: Every High Court has many other judges.
  Fact 0.2: The number of other judges in a High Court is defined by the President of India.

Retrieved items:
<item 1>
List of Justices of the Supreme Court of Sri Lanka
Justice of the Straits Settlements 35 George Henry Withers -- -- -- 36 Dodwell Browne -- -- -- 37 Fredrick Charles Moncrieff 4 April 1900 -- -- -- 38 Henry
Lorentz Wendt 15 October 1901 -- 28 November 1910 -- -- 39 John Page Middleton 28 January 1902 -- 1912 -- -- Puisine Judge of the Supreme Court of Cyprus 40
Charles Layard Western 26 April 1902 -- 18 June 1906 26 April 1902 -- 18 June 1906 1,514 1,514 41 Joseph Richard Grenier 28 November 1910 -- 1912 -- -- 42
Alexander Wood Renton 3 September 1905 -- 1918 22 August 1914 - 1918 -- 43 Alfred George Lascelles England 1 May 1911 -- 1914 1 May 1911 -- 1914 -- 44 Joseph
Turner Hutchinson England 23 October 1906 -- 1 May 1911 23 October 1906 -- 1 May 1911 1,
</item 1>

<item 2>
Indian High Courts Act 1861
Qualifications for the Judges ( a ) He should be a citizen of India . ( b ) He should have been ( I ) a judge for 10 years of Subordinate court under the
Judicial Service of the State or ( ii ) an Advocate for 10 years in a High Courts in India ( Article 217 ) . Tenure : Originally the age of the retirement of
the judges of the High Courts was fixed at 60 but it was raised to 62 in 1963 according to the 15th amendment of the Constitution . Removal of the Judges : A
judge may leave his office by resigning . He will send his letter of resignation to the President . His office would be considered to have been vacated if he is
appointed as a judge of the Supreme Court or is transferred to some other High Court . A judge of a High Court may also be removed like a judge of the Supreme
Court . A judge of High Court may be removed by the President if the Parliament passes a motion against him by an absolute majority and 2 / 3rd majority of the
members present and voting , both the Houses sitting separately .
</item 2>

<item 3>
List of Justices of the Supreme Court of Sri Lanka
Imam Central 9 January 2009 -- 20 February 2013 -- 1,503 -- Rajapaksa Judge of the Court of Appeal R.K.S. Suresh Chandra 17 June 2010 -- -- -- Rajapaksa Sathya
Hettige 10 July 2011 -- 18 April 2014 -- 1,013 -- Rajapaksa University of Colombo President of the Court of Appeal of Sri Lanka Priyasath Dep † 10 July 2011 --
present 02 March 2017 - present -- 7003248200000000000 ♠ 2,482 -- Rajapaksa University of Colombo International Institute of Social Studies Solicitor General of
Sri Lanka Eva Wansundera † North Western 7 July 2012 -- present -- 7003211900000000000 ♠ 2,
</item 3>

<item 4>
Indian High Courts Act 1861
Salary : The pay of the Chief Justice of a High Court is rupees 90,000 / - per month and that of the other judges is rupees 85,000 / - per month . Powers and
Functions Original Jurisdiction : The original jurisdiction of the High Court is restricted . ( a ) Every High Court under Article 226 is empowered to issue
writs , orders , directions including writs in the nature of Habeas Corpus , Mandamus , Prohibition , Quo - warranto and Certiorari or any of them to any person
or authority with in its territory for the enforcement of the Fundamental Rights and for any other purpose . ( b ) The original jurisdiction of High Court
extends to matters of admiralty , matrimonial , contempt of court and cases ordered to be transferred to High Court by lower court . ( c ) The High Courts of
Mumbai , Kolkata and Chennai have original jurisdiction on hearing straightway cases involving the Christians and Parsies . ( d ) The High Courts of Mumbai ,
Kolkata and Chennai exercise original civil jurisdiction when the amount involved is more than two thousand rupees .
</item 4>

<item 5>
Indian High Courts Act 1861
Appellate Jurisdiction : The appellate jurisdiction of the High Courts extends so : ( a ) The High Court can hear appeals in civil cases if the amount involved
in the case is at least Rs. 5000 . ( b ) The High Court in criminal cases hears the appeal in which the accused has been sentenced to four years imprisonment by
the Sessions Judge . v ( c ) The death sentence awarded by Sessions Judge is subject to approval by the High Court . ( d ) The High Court hear the cases
involving interpretation of the Constitution or Law . ( e ) The High Court hears the cases on income tax , sales tax etc . Power of Judicial Review : The States
High Courts like the Supreme Court has the power of Judicial Review . A High Court has the power to strike down any law of the State or any order of the
executive if it violates any provision of the constitution or curtails or takes any of the Fundamental Rights of the people . Administrative and Supervisory
Power : The State High Court performs many administrative functions within its Territorial Jurisdiction .
</item 5>

<item 6>
Indian High Courts Act 1861
It exercises the power of superintendence and control over all courts and tribunals throughout the territory except the military tribunals . Judicial tenure and
seniority ( edit ) Under § 5 of the Act , judges served at the pleasure of Her Majesty . The chief justice had precedence over judges , whereas judges had
seniority based on appointment . Under § 14 , the chief justice had the power to form divisions of two or more judges to hear cases . Jurisdiction ( edit )
Under article 9 of the Act , each High Court had `` all such powers and authority for and in relation to the administration of justice '' including original and
appellate jurisdiction over civil , criminal , admiralty , vice-admiralty , testamentary , intestate , and matrimonial matters . References ( edit ) Jump up ^
`` Indian High Courts Act 1861 '' . GKToday. 2011 - 10 - 17 . Retrieved 2017 - 03 - 20 . Jump up ^ Criminal Justice India Series : Punjab , 2002 . Allied
Publishers . 2002 . p. 233 .
</item 6>

<item 7>
Indian High Courts Act 1861
The High Court of a State is the highest court of the State and all other courts of the State work under it . Normally there is one High Court in every State
but there can be only one High Court for two or more States as well , according to the constitution . There is one High Court at Chandigarh for Punjab , Haryana
and Union Territory of Chandigarh . Similarly there is one High Court at Guwahati which serves Assam , Arunachal Pradesh , Mizoram and Nagaland . Composition :
In every High Court , there is a Chief Justice and many other judges whose number is defined by the President of India . Appointment of the Judges : The Chief
Justice of a High Court is appointed by the President with the consultation of the Chief Justice of the Supreme Court and the Governor of the State . The other
judges are appointed by the will of President , Governor and the Chief Justice of High Court . Qualifications for the Judges ( a ) He should be a citizen of
India .
</item 7>

<item 8>
Justice of the peace
Until the Courts Act 2003 came into force , magistrates were tied to a particular area ( see magistrates ' courts committee , commission area , petty sessions
area ) . This has now been changed such that they are assigned to local justice areas , but less strongly . The Courts Act provides the current framework for
appointment of the justices , which is done by the Lord Chancellor in the name of sovereign . Justices can also be removed by the same mechanism . District
judges ( magistrates ' court ) -- previously known as stipendiary magistrates -- must have a seven years ' general legal qualification , and are appointed by
the sovereign on the advice of the Lord Chancellor . Before 1714 , magistrates were liable to be approached at any time and in any place by people legally
recognised as paupers , appealing for aid if parish authorities refused to provide any . It was relatively common for these magistrates to write out , on the
spot , an order requiring aid to be granted . Scotland ( edit ) See also : Judiciary of Scotland Within the Scottish legal system justices of the peace are lay
magistrates who currently sit in the justice of the peace courts .
</item 8>

<item 9>
Indian High Courts Act 1861
Contents ( hide ) 1 Abolishing Existing Courts 2 Qualifications of High Court Judges 3 Judicial Tenure and Seniority 4 Jurisdiction 5 References Abolishing
existing Courts ( edit ) The Act abolished the Supreme Courts at Calcutta , Madras , and Bombay ; the Sadar Diwani Adalat and the Sadar Nizamat Adalat at
Calcutta ; Sadar Adalat and Faujdari Adalat at Madras ; Sadar Diwani Adalat and Faujdari Adalat at Bombay ( § 8 ) . Qualifications of High Court judges ( edit )
Each High Court could consist of a chief justice and up to 15 judges . Under § 3 of the Act , judges could be selected from barristers ( with 5 years of
experience ) , civil servants ( with 10 years of experience including 3 years as a zillah judge ) , judges of small cause courts or sudder ameen ( with 5 years
of experience ) , or pleaders of lander courts or High Courts ( with 05 years of experience ) .
</item 9>

<item 10>
List of Justices of the Supreme Court of Sri Lanka
182 -- Sirisena Sri Lanka Law College University of Exeter Judge of the Court of Appeal Kankanithanthri Chitrasiri † 3 December 2015 -- present --
7002875000000000000 ♠ 875 -- Sirisena University of Sri Lanka Queen Mary University of London Judge of the Court of Appeal Nalin Perera † 3 March 2016 --
present -- 7002784000000000000 ♠ 784 -- Sirisena Judge of the Court of Appeal Prasanna Sujeewa Jayawardena † 16 June 2016 -- present -- 7002679000000000000 ♠
679 -- Sirisena University of Colombo Judge of the Court of Appeal See also ( edit ) List of Chief Justices of Sri Lanka List of Justices of the Supreme Court
of Sri Lanka by court composition Notes ( edit ) Jump up ^ Provisional Chief Justice Jump up ^ On leave ( 1 January - 28 April 1871 ) Jump up ^ Acting Chief
Justice ( 1 - 10 October 1879 ,
</item 10>
=== END ITEM id=85022bfb40766a99 ===

=== ITEM id=b5c8074c9d6a06ea ===
Gold sentences and facts:
Sentence 0: Kylo Ren is a fictional character in the Star Wars franchise . Introduced in the 2015 film Star Wars : The Force Awakens , he is portrayed by Adam Driver . Kylo Ren is the chosen name of Ben Solo , the son of original Star Wars trilogy characters Han Solo and Leia Organa . Though trained by his uncle Luke Skywalker as a Jedi , he has been seduced to the dark side of the Force by Supreme Leader Snoke and aspires to be as powerful as his grandfather , Darth Vader . Kylo Ren is also the master of the Knights of Ren , as well as a commander and later the supreme leader of the First Order , an organization spawned from the fallen Galactic Empire . He is featured in The Force Awakens media and merchandising and appears in the film 's sequel , Star Wars : The Last Jedi ( 2017 ) .
  Fact 0.0: Kylo Ren is a fictional character in the Star Wars franchise.
  Fact 0.1: Kylo Ren was introduced in the 2015 film Star Wars: The Force Awakens.
  Fact 0.2: Kylo Ren is portrayed by Adam Driver.
  Fact 0.3: Kylo Ren is the chosen name of Ben Solo.
  Fact 0.4: Ben Solo is the son of original Star Wars trilogy characters Han Solo and Leia Organa.
  Fact 0.5: Kylo Ren was trained by his uncle Luke Skywalker as a Jedi.
  Fact 0.6: Kylo Ren has been seduced to the dark side of the Force by Supreme Leader Snoke.
  Fact 0.7: Kylo Ren aspires to be as powerful as his grandfather, Darth Vader.
  Fact 0.8: Kylo Ren is the master of the Knights of Ren.
  Fact 0.9: Kylo Ren is a commander of the First Order.
  Fact 0.10: Kylo Ren later became the supreme leader of the First Order.
  Fact 0.11: The First Order is an organization spawned from the fallen Galactic Empire.
  Fact 0.12: Kylo Ren is featured in The Force Awakens media and merchandising.
  Fact 0.13: Kylo Ren appears in the film's sequel, Star Wars: The Last Jedi.
  Fact 0.14: Star Wars: The Last Jedi was released in 2017.
Sentence 1: Abrams told Empire in August 2015 , `` Kylo Ren is not a Sith . He works under Supreme Leader Snoke , who is a powerful figure on the Dark Side of the Force . '' Abrams had previously told Entertainment Weekly that the character , `` came to the name Kylo Ren when he joined a group called the Knights of Ren . '' Robbie Collin of The Telegraph described Ren as `` a hot - headed , radicalised Dark Side jihadi , whose red lightsaber splutters and crackles as violently as his temper '' . Abrams noted , `` The lightsaber is something that he built himself , and is as dangerous and as fierce and as ragged as the character . '' The Telegraph also explains that Ren 's wild and erratic temper and `` angsty '' instability make him dangerous . Melissa Leon of The Daily Beast describes Ren 's use of the Force as `` formidable '' , citing his ability to stop a blaster shot mid-air , immobilize victims and probe their minds against their will .
  Fact 1.0: Abrams told Empire in August 2015 that Kylo Ren is not a Sith.
  Fact 1.1: Abrams told Empire in August 2015 that Kylo Ren works under Supreme Leader Snoke.
  Fact 1.2: Abrams told Empire in August 2015 that Supreme Leader Snoke is a powerful figure on the Dark Side of the Force.
  Fact 1.3: Abrams had previously told Entertainment Weekly that the character came to the name Kylo Ren when he joined a group called the Knights of Ren.
  Fact 1.4: Robbie Collin of The Telegraph described Ren as a hot-headed, radicalised Dark Side jihadi.
  Fact 1.5: Robbie Collin of The Telegraph described Ren's red lightsaber as spluttering and crackling as violently as his temper.
  Fact 1.6: Abrams noted that the lightsaber is something Ren built himself.
  Fact 1.7: Abrams noted that the lightsaber is as dangerous, fierce and ragged as the character.
  Fact 1.8: The Telegraph explains that Ren's wild and erratic temper and angsty instability make him dangerous.
  Fact 1.9: Melissa Leon of The Daily Beast describes Ren's use of the Force as formidable.
  Fact 1.10: Melissa Leon cites Ren's ability to stop a blaster shot mid-air.
  Fact 1.11: Melissa Leon cites Ren's ability to immobilize victims.
  Fact 1.12: Melissa Leon cites Ren's ability to probe victims' minds against their will.

Retrieved items:
<item 1>
Kylo Ren
You normally think of order , and structure , and full commitment and no hesitation ... he 's just a little bit more unpolished . It 's in his costume , in his
lightsaber -- how you kind of get the sense that it could just not work at any moment ; that it could just blow up . That 's kind of like a big metaphor for him
. Appearances ( edit ) Model of Kylo 's cross-bladed lightsaber on display at Star Wars Launch Bay at Disney 's Hollywood Studios ( 2015 ) . Feature films (
edit ) The Force Awakens ( 2015 ) ( edit ) Main article : Star Wars : The Force Awakens Kylo Ren first appears in The Force Awakens as the commander of the
First Order , a sinister regime that has risen from the remains of the old Galactic Empire . After arriving at Jakku to retrieve a map containing the
coordinates for the location of Luke Skywalker , Ren kills an old vicar known as Lor San Tekka and captures Resistance pilot Poe Dameron , who has also been
sent to recover the map by General Leia Organa .
</item 1>

<item 2>
Kylo Ren
Kylo Ren - wikipedia Kylo Ren Jump to : navigation , search `` Kylo '' redirects here . For the web browser , see Kylo ( web browser ) . Kylo Ren Ben Solo Star
Wars character Kylo Ren 's costume displayed on a mannequin First appearance The Force Awakens ( 2015 ) Created by Lawrence Kasdan J.J.
</item 2>

<item 3>
Kylo Ren
Before the duel is finished , they are separated by a seismic fissure created by the collapsing base . Rey and the others escape as Snoke orders General Hux to
evacuate the base and bring Ren to him to complete his training . The last Jedi ( 2017 ) ( edit ) Main article : Star Wars : The Last Jedi Ren 's inner conflict
continues into The Last Jedi , specifically through his conversations with Rey , with whom he connects through the Force . Rey believes that his conflict will
ultimately bring him back to the light side . Ren captures her and brings her before Snoke , who orders Ren to kill her . However , Ren uses the Force to ignite
a lightsaber at Snoke 's side and cut him in half , killing him . Ren beseeches Rey to join him in creating a new order , separate from the legacies created by
Snoke and Luke . When Rey refuses , Ren declares himself Supreme Leader of the First Order , and orders his forces to attack the Resistance base on Crait .
</item 3>

<item 4>
Kylo Ren
Though trained by his uncle Luke Skywalker as a Jedi , he has been seduced to the dark side of the Force by Supreme Leader Snoke and aspires to be as powerful
as his grandfather , Darth Vader . Kylo Ren is also the master of the Knights of Ren , as well as a commander and later the supreme leader of the First Order ,
an organization spawned from the fallen Galactic Empire . He is featured in The Force Awakens media and merchandising and appears in the film 's sequel , Star
Wars : The Last Jedi ( 2017 ) . Contents ( hide ) 1 Concept and creation 2 Character 3 Appearances 3.1 Feature films 3.1. 1 The Force Awakens ( 2015 ) 3.1. 2
The Last Jedi ( 2017 ) 3.2 Related works and promotional material 3.3 Novels 4 Reception 5 Family tree 6 References 7 External links Concept and creation ( edit
) After Lucasfilm was acquired by The Walt Disney Company in 2012 , the script for The Force Awakens was written by Lawrence Kasdan , J.J.
</item 4>

<item 5>
Kylo Ren
Ren engages Luke in battle and appears to blow him up and cut him in half , but Luke remains standing ; `` Luke '' is revealed to be simply a vision projected
from the real Luke on Ahch - To , serving as a distraction which allows the Resistance to escape from the First Order . When Luke vanishes , the First Order
storms the base , but Resistance has already evacuated ; he shares a final look with Rey through the Force before Rey closes the door to the Millennium Falcon
and escapes with the Resistance . Related works and promotional material ( edit ) He is a playable character in the 2015 Force Awakens add - on to the Disney
Infinity 3.0 video game , with an Infinity character figurine available separately . He is also a character in the strategy video game Star Wars : Force Arena .
Hasbro has released a 3 ⁄ - inch ( 9.5 cm ) Kylo Ren action figure , and a 6 - inch ( 15 cm ) figure in their Black Series line .
</item 5>

<item 6>
Kylo Ren
Abrams Michael Arndt Portrayed by Adam Driver Voiced by Adam Driver ( Disney Infinity 3.0 , Lego Star Wars : The Force Awakens , Lego Star Wars : The Resistance
Rises ) Matthew Wood ( Star Wars Battlefront II ) Roger Craig Smith ( Star Wars Battlefront II ) Information Full name Ben Solo Species Human Gender Male
Occupation Supreme Leader of the First Order Master of the Knights of Ren Former Jedi Padawan Affiliation First Order Knights of Ren New Jedi Order ( formerly )
Family Han Solo ( father ) Leia Organa ( mother ) Relatives Luke Skywalker ( maternal uncle ) Anakin Skywalker ( maternal grandfather ) Padmé Amidala ( maternal
grandmother ) Bail Organa ( adopted maternal grandfather ) Breha Organa ( adopted maternal grandmother ) Shmi Skywalker ( maternal great - grandmother )
Homeworld Chandrila Kylo Ren is a fictional character in the Star Wars franchise . Introduced in the 2015 film Star Wars : The Force Awakens , he is portrayed
by Adam Driver . Kylo Ren is the chosen name of Ben Solo , the son of original Star Wars trilogy characters Han Solo and Leia Organa .
</item 6>

<item 7>
Kylo Ren
Novels ( edit ) Kylo Ren is a point of view character in the 2015 novelization of The Force Awakens by Alan Dean Foster . In the 2017 Chuck Wendig novel Star
Wars : Aftermath : Empire 's End , Ben Solo is stated as having been born on the planet Chandrila on the same day as a peace treaty is signed between the
remnants of the Empire and the New Galactic Republic . Kylo Ren is additionally mentioned in the 2017 novel Star Wars : Phasma . Reception ( edit ) Adam Driver
received positive reviews for his performance as Kylo Ren . Both the character of Kylo Ren and Driver 's performance have been positively received . Many
complimented Ren 's conflicted nature and depth , as well as his costume design , and noted there were many places the character could be taken in future
installments . Peter Bradshaw of The Guardian praised the character and the actor alike , saying , `` He is gorgeously cruel , spiteful and capricious -- and
unlike the Vader of old , he is given to petulant temper tantrums , with his lightsaber drawn . ''
</item 7>

<item 8>
Kylo Ren
Abrams and Michael Arndt , with no direct participation from Star Wars creator George Lucas . According to director Abrams , the First Order is inspired by
ODESSA , a theoretical network of SS officers who fled to Argentina following World War II . Driver 's casting in the film in an unnamed role was first
announced on April 29 , 2014 . Kylo Ren was first seen , but still not named , in the 88 - second The Force Awakens teaser trailer released by Lucasfilm on
November 28 , 2014 , wielding a jagged red lightsaber with a crossguard . The name Kylo Ren was revealed by Entertainment Weekly in a Lucasfilm - designed Topps
- style trading card mockup in December 2014 . A May 2015 Vanity Fair photo shoot by Annie Leibovitz confirmed that Driver would be portraying Kylo . In an
interview with Time , The Force Awakens costume designer Michael Kaplan said that Abrams requested that Ren 's mask be designed to be memorable for a child .
After numerous attempts to gain Abrams ' approval , the final design was selected .
</item 8>

<item 9>
Kylo Ren
Ren 's character arc shares similarities with that of the Star Wars expanded universe character Jacen Solo , the son of Han Solo and Princess Leia who threatens
the galaxy as a fallen Jedi . Additionally , critics have noted a resemblance between Ren 's character design and that of Revan , the protagonist of Knights of
the Old Republic . Character ( edit ) Abrams told Empire in August 2015 , `` Kylo Ren is not a Sith . He works under Supreme Leader Snoke , who is a powerful
figure on the Dark Side of the Force . '' Abrams had previously told Entertainment Weekly that the character , `` came to the name Kylo Ren when he joined a
group called the Knights of Ren . '' Robbie Collin of The Telegraph described Ren as `` a hot - headed , radicalised Dark Side jihadi , whose red lightsaber
splutters and crackles as violently as his temper '' . Abrams noted , `` The lightsaber is something that he built himself , and is as dangerous and as fierce
and as ragged as the character . '' The Telegraph also explains that Ren 's wild and erratic temper and `` angsty '' instability make him dangerous .
</item 9>

<item 10>
Kylo Ren
Ren soon learns that the pilot had entrusted his astromech droid , BB - 8 , with the map . Poe flees with the help of rogue stormtrooper Finn , who later finds
BB - 8 , and the scavenger Rey . Finn , Rey , and BB - 8 escape Jakku in the Millennium Falcon , and are soon intercepted by the ship 's former owner , Han Solo
, and his co-pilot Chewbacca . It is revealed that Ren is the son of Han and Leia , and was once one of Luke 's Jedi pupils . He was drawn to the dark side of
the Force by the First Order 's Supreme Leader Snoke . He then helped to destroy his uncle Luke 's new Jedi Academy . However , Ren still feels the pull of the
light side of the Force and seeks the strength to overcome this weakness from his grandfather Darth Vader , whose burnt helmet is in Ren 's possession .
Arriving at the Maz Kanata castle , Ren captures Rey , who he senses has seen the map . While interrogating her with the Force , he realizes that she is strong
with the Force herself , though unaware of it .
</item 10>
=== END ITEM id=b5c8074c9d6a06ea ===

=== ITEM id=e97bdc2e9f7cb5ac ===
Gold sentences and facts:
Sentence 0: Fortnite is set in contemporary Earth , where the sudden appearance of a worldwide storm causes 98 % of the world 's population to disappear , and zombie - like creatures rise to attack the remainder . Considered by Epic as a cross between Minecraft and Left 4 Dead , Fortnite has up to four players cooperating on various missions on randomly - generated maps to collect resources , build fortifications around defensive objectives that are meant to help fight the storm and protect survivors , and construct weapons and traps to engage in combat with waves of these creatures that attempt to destroy the objectives . Players gain rewards through these missions to improve their hero characters , support teams , and arsenal of weapon and trap schematics to be able to take on more difficult missions . The game is supported through microtransactions to purchase in - game currency that can be used towards these upgrades .
  Fact 0.0: Fortnite is set in contemporary Earth.
  Fact 0.1: A sudden appearance of a worldwide storm causes 98% of the world's population to disappear.
  Fact 0.2: Zombie-like creatures rise to attack the remainder of the population.
  Fact 0.3: Epic considers Fortnite a cross between Minecraft and Left 4 Dead.
  Fact 0.4: Fortnite has up to four players cooperating on various missions.
  Fact 0.5: The missions are on randomly generated maps.
  Fact 0.6: Players collect resources in the missions.
  Fact 0.7: Players build fortifications around defensive objectives.
  Fact 0.8: The fortifications are meant to help fight the storm and protect survivors.
  Fact 0.9: Players construct weapons and traps.
  Fact 0.10: Players engage in combat with waves of zombie-like creatures.
  Fact 0.11: The creatures attempt to destroy the objectives.
  Fact 0.12: Players gain rewards through the missions.
  Fact 0.13: Rewards are used to improve hero characters, support teams, and arsenal of weapon and trap schematics.
  Fact 0.14: The rewards help players take on more difficult missions.
  Fact 0.15: The game is supported through microtransactions.
  Fact 0.16: Microtransactions purchase in-game currency.
  Fact 0.17: In-game currency can be used towards these upgrades.
Sentence 1: Fortnite was first revealed at the 2011 Spike Video Game Awards ( VGA ) , with Epic 's former design director Cliff Bleszinski introducing a trailer for the game . Donald Mustard , creative lead at Epic , said in 2017 that this announcement was `` three weeks after we came up with the idea , before we even made the game '' . The title , which started out as an internal game jam project , represents a departure from the company 's previous work . As Bleszinski explained during the Spike event , Epic wanted to `` switch things up a little bit and do something different and fun '' with Fortnite , describing it as `` a world where you explore , you scavenge , you build and ultimately you survive . '' In an interview with Engadget , he also echoed these statements , claiming that the game would be different from the Gears of War series : `` There 's no dudebros in it ... Not that there 's anything wrong with that , right ? But creatively for the team , Gears has been amazing for us . But it 's fun to kind of stretch our wings and do something that 's a little different from the usual . '' Epic Games Vice President of Publishing Mike Fischer said in 2015 that Epic recognized that they `` announced this game too soon '' , and that its lengthy development period was due to `` very good reasons . ''
  Fact 1.0: Fortnite was first revealed at the 2011 Spike Video Game Awards (VGA).
  Fact 1.1: Epic's former design director Cliff Bleszinski introduced a trailer for the game at the 2011 Spike Video Game Awards.
  Fact 1.2: Donald Mustard, creative lead at Epic, said in 2017 that the announcement was three weeks after we came up with the idea, before we even made the game.
  Fact 1.3: Fortnite started out as an internal game jam project.
  Fact 1.4: Fortnite represents a departure from the company's previous work.
  Fact 1.5: Bleszinski said Epic wanted to switch things up a little bit and do something different and fun with Fortnite.
  Fact 1.6: Bleszinski described Fortnite as a world where you explore, you scavenge, you build and ultimately you survive.
  Fact 1.7: In an interview with Engadget, Bleszinski claimed the game would be different from the Gears of War series.
  Fact 1.8: Bleszinski said there are no dudebros in Fortnite.
  Fact 1.9: Bleszinski said Gears has been amazing for the team.
  Fact 1.10: Bleszinski said it is fun to stretch the team's wings and do something a little different from the usual.
  Fact 1.11: Epic Games Vice President of Publishing Mike Fischer said in 2015 that Epic recognized they announced this game too soon.
  Fact 1.12: Mike Fischer said the lengthy development period was due to very good reasons.

Retrieved items:
<item 1>
Fortnite
Fortnite - wikipedia Fortnite Jump to : navigation , search For the battle royale version of the game , see Fortnite Battle Royale . Fortnite European box art
Developer ( s ) Epic Games People Can Fly Publisher ( s ) Epic Games Director ( s ) Darren Sugg Artist ( s ) Pete Ellis Composer ( s ) Rom Di Prisco Engine
Unreal Engine 4 Platform ( s ) Microsoft Windows macOS PlayStation 4 Xbox One iOS Release 2018 Genre ( s ) Survival Mode ( s ) Single - player , multiplayer
Fortnite is a co-op sandbox survival game developed by Epic Games and People Can Fly and published by Epic Games . The game was released as a paid - for early
access title for Microsoft Windows , macOS , PlayStation 4 and Xbox One on July 25 , 2017 , with a full free - to - play release expected in 2018 . The retail
versions of the game were published by Gearbox Publishing , while online distribution of the PC versions is handled by Epic 's launcher . Fortnite is set in
contemporary Earth , where the sudden appearance of a worldwide storm causes 98 % of the world 's population to disappear , and zombie - like creatures rise to
attack the remainder .
</item 1>

<item 2>
Fortnite
To help with this transition , the company brought in Tencent who had several games operating under this model . As part of this agreement , Tencent bought a
significant share of ownership in Epic Games through stock acquisition around June 2012 . The shifting of Epic 's approach through Tencent led to the departure
of some high - level staff later that year , including Bleszinski . Fortnite was seen as the spearhead for Epic 's games - as - a-service model as they
collaborated with Tencent . This caused some road bumps in the development of Fortnite , according to Mustard . Fortnite 's development was also slowed as it
was used as the testing ground for the new features of the Unreal 4 engine . By November 2013 , Epic confirmed that Fortnite would not release that year , nor
offered a target released date , though affirmed the game was still in development by several of its studios . Fortnite was a feature in the May 2014 issue of
Game Informer , revealing that the title would be released as a free - to - play game . By 2014 , Fortnite was at a `` pretty functional prototype '' with most
of the Unreal 4 engine elements smoothed out , according to Mustard .
</item 2>

<item 3>
Fortnite
Epic Games Vice President of Publishing Mike Fischer said in 2015 that Epic recognized that they `` announced this game too soon '' , and that its lengthy
development period was due to `` very good reasons . '' As the game was at its very preliminary stages at the VGA reveal , the goal of this reveal was to seek
public interest in the title and potential publishing partners as to decide on the game 's release platforms and timeframe . During the July 2012 San Diego
Comic Con , Epic announced that Fortnite would be an exclusive personal computer title , and the first one to be developed by Epic using their new Unreal 4 game
engine , with a planned release in 2013 . The game 's development was originally started in the Unreal 3 engine , but as they progressed , they had seen the
opportunity to work in several of the new feature sets and scripting language offered by Unreal 4 for Fortnite , while still running on most personal computers
at that time .
</item 3>

<item 4>
Fortnite
Jump up ^ Brown , Fraser ( December 10 , 2014 ) . `` Epic pulls back the curtain ( and flesh ) on Fortnite 's monster design '' . PCGamesN . Archived from the
original on July 30 , 2017 . Retrieved July 29 , 2017 . Jump up ^ Greening , Chris ( October 22 , 2016 ) . `` Game Releases Calendar : 2016 Edition '' . Game
Music Online . Archived from the original on September 30 , 2017 . Retrieved July 16 , 2017 . Jump up ^ McWhertor , Michael ( July 12 , 2012 ) . `` Epic Games '
' Fortnite ' will be the developer 's first Unreal Engine 4 game '' . Polygon . Archived from the original on July 30 , 2017 . Retrieved August 1 , 2017 . Jump
up ^ Makuch , Eddie ( July 12 , 2013 ) . `` Epic : Fortnite is `` Minecraft meets Left 4 Dead '' `` . GameSpot . Archived from the original on March 27 , 2015 .
</item 4>

<item 5>
Fortnite
`` Epic Games ' New Franchise FORTnITE Blends Survival Horror With Tower Defense Strategy '' . Forbes . Archived from the original on September 30 , 2017 .
Retrieved July 24 , 2017 . Jump up ^ Markovech , Sam ( June 8 , 2017 ) . `` Fortnite 's years of delays end with not - free - to - play version coming in July
'' . Ars Technica . Archived from the original on June 8 , 2017 . Retrieved June 8 , 2017 . ^ Jump up to : McWhertor , Michael ( March 26 , 2014 ) . `` What 's
the future of games at Epic Games ? '' . Polygon . Archived from the original on June 27 , 2017 . Retrieved June 9 , 2017 . Jump up ^ Schramm , Mike ( December
11 , 2011 ) . `` Cliff B talks Fortnite : ' There 's no dudebros in it ' '' . Engadget . Archived from the original on October 1 , 2017 .
</item 5>

<item 6>
Fortnite
Engadget . Archived from the original on October 1 , 2017 . Retrieved July 24 , 2017 . Jump up ^ Orland , Kyle ( June 8 , 2015 ) . `` Hands - on : Fortnite is
an overwhelming zombie defense experience '' . Ars Technica . Archived from the original on December 8 , 2017 . Retrieved December 7 , 2017 . Jump up ^ Yin -
Pool , Wesley ( December 12 , 2011 ) . `` Epic reveals Minecraft inspiration for Fortnite '' . Eurogamer . Archived from the original on December 1 , 2017 .
Retrieved November 29 , 2017 . Jump up ^ Rigney , Ryan ( July 12 , 2012 ) . `` First Screens : Epic 's Fortnite Is the First Unreal Engine 4 Game '' . Wired .
Archived from the original on December 22 , 2016 . Retrieved November 29 , 2017 . Jump up ^ Hafer , T.J. ( July 12 , 2012 ) .
</item 6>

<item 7>
Fortnite
They further opted for personal computer exclusivity to avoid the difficulty of having to go through console certification , and as they planned to be
constantly monitoring and tweaking the game , acting as a dungeon master , the personal computer approach would allow them to do this without restrictions
normally set by console manfuctures . Bleszinski later clarified that they would not rule out release on other platforms as they developed the title . Fortnite
cosplayers at Gamescom 2017 Fortnite 's development was spread among several of Epic 's satellite studios , and was also co-developed by the Polish studio
People Can Fly , which had worked with Epic previously on earlier games , and had been fully acquired by Epic sometime in 2012 . People Can Fly were briefly
renamed Epic Games Poland in 2013 as to align with Epic 's other studios . By March 2014 , there were about 90 developers working on the game . People Can Fly
later returned to being an independent studio and their own name in 2015 , but continued to help Epic with Fortnite 's development . During Fortnite 's
development , Epic had seen the game industry shifting to games as a service model .
</item 7>

<item 8>
Fortnite
Development History Fortnite was first revealed at the 2011 Spike Video Game Awards ( VGA ) , with Epic 's former design director Cliff Bleszinski introducing a
trailer for the game . Donald Mustard , creative lead at Epic , said in 2017 that this announcement was `` three weeks after we came up with the idea , before
we even made the game '' . The title , which started out as an internal game jam project , represents a departure from the company 's previous work . As
Bleszinski explained during the Spike event , Epic wanted to `` switch things up a little bit and do something different and fun '' with Fortnite , describing
it as `` a world where you explore , you scavenge , you build and ultimately you survive . '' In an interview with Engadget , he also echoed these statements ,
claiming that the game would be different from the Gears of War series : `` There 's no dudebros in it ... Not that there 's anything wrong with that , right ?
But creatively for the team , Gears has been amazing for us . But it 's fun to kind of stretch our wings and do something that 's a little different from the
usual . ''
</item 8>

<item 9>
Fortnite
( July 12 , 2012 ) . `` Fortnite will be a PC exclusive and Epic 's first Unreal Engine 4 game '' . PC Gamer . Archived from the original on December 1 , 2017 .
Retrieved November 29 , 2017 . Jump up ^ McGee , Maxwell ( July 13 , 2012 ) . `` Epic 's Cliff Bleszinski and Tanya Jessen Talk Up Fortnite '' . GameSpot .
Archived from the original on February 6 , 2018 . Retrieved November 29 , 2017 . ^ Jump up to : Purchase , Robert ( September 1 , 2012 ) . `` Fortnite detailed
at PAX Prime ; Minecraft meets DayZ '' . Eurogamer . Archived from the original on December 1 , 2017 . Retrieved November 29 , 2017 . Jump up ^ Sheridan ,
Connor ( November 1 , 2013 ) . `` People Can Fly working on Fortnite , renamed Epic Games Poland '' . GamesRadar . Archived from the original on December 2 ,
2017 .
</item 9>

<item 10>
Fortnite
Fortnite was being developed alongside Paragon , which Epic announced on November 2015 . As Paragon seemed to take Epic 's focus , leaving little news about
Fortnite , CEO Tim Sweeney said in March 2016 that they were still committed to Fortnite once Paragon was launched and established , given that much of the work
on Fortnite would take time to get the right balance for gameplay . `` We figure we should start with one major successful launch and do one at a time .
Fortnite will be next . '' By June 2017 , Epic Games announced that Fortnite was now set for a 2018 release across Windows , macOS , and the PlayStation 4 and
Xbox One consoles . Leading up to this free - to - play release , the game was offered as a paid early access period starting on July 25 , 2017 for all
platforms ; players who pre-ordered Founder 's Packs were granted access to the game on July 21 . The lengthy period since the game 's alpha phases was ascribed
to developing Fortnite as a games - as - a-service model , according to creative lead Donald Mustard .
</item 10>
=== END ITEM id=e97bdc2e9f7cb5ac ===

=== ITEM id=e7b256f5a398e61c ===
Gold sentences and facts:
Sentence 0: In healthy adults , there are two normal heart sounds , often described as a lub and a dub ( or dup ) , that occur in sequence with each heartbeat . These are the first heart sound ( S ) and second heart sound ( S ) , produced by the closing of the atrioventricular valves and semilunar valves , respectively . In addition to these normal sounds , a variety of other sounds may be present including heart murmurs , adventitious sounds , and gallop rhythms S and S .
  Fact 0.0: In healthy adults there are two normal heart sounds.
  Fact 0.1: The two normal heart sounds are often described as a lub and a dub (or dup).
  Fact 0.2: The two normal heart sounds occur in sequence with each heartbeat.
  Fact 0.3: The first heart sound (S) is produced by the closing of the atrioventricular valves.
  Fact 0.4: The second heart sound (S) is produced by the closing of the semilunar valves.
  Fact 0.5: The closing of the atrioventricular valves and the semilunar valves produces the first and second heart sounds, respectively.
  Fact 0.6: In addition to the normal sounds, a variety of other sounds may be present.
  Fact 0.7: The other sounds that may be present include heart murmurs, adventitious sounds, and gallop rhythms labeled S and S.

Retrieved items:
<item 1>
Heart sounds
Murmurs may be physiological ( benign ) or pathological ( abnormal ) . Abnormal murmurs can be caused by stenosis restricting the opening of a heart valve ,
resulting in turbulence as blood flows through it . Abnormal murmurs may also occur with valvular insufficiency ( regurgitation ) , which allows backflow of
blood when the incompetent valve closes with only partial effectiveness . Different murmurs are audible in different parts of the cardiac cycle , depending on
the cause of the murmur . Contents ( hide ) 1 Primary heart sounds 2 Extra heart sounds 3 Murmurs 3.1 Effects of breathing 3.2 Interventions that change murmurs
4 Other abnormal sounds 5 Surface anatomy 6 Recording heart sounds 7 See also 8 Notes and references 9 External links Primary heart sounds ( edit ) Normal heart
sounds are associated with heart valves closing : S ( edit ) The first heart sound , or S , forms the `` lub '' of `` lub - dub '' and is composed of components
M ( mitral valve closure ) and T ( tricuspid valve closure ) . Normally M precedes T slightly .
</item 1>

<item 2>
Heart sounds
Heart sounds - wikipedia Heart sounds Jump to : navigation , search Normal heart sounds Normal heart sounds as heard with a stethoscope Problems playing this
file ? See media help . 72 BPM Heart sound Front of thorax , showing surface relations of bones , lungs ( purple ) , pleura ( blue ) , and heart ( red outline )
. The locations of best auscultation for each heart valve are labeled with `` M '' , `` T '' , `` A '' , and `` P '' . First heart sound : caused by
atrioventricular valves - Mitral ( M ) and Tricuspid ( T ) . Second heart sound caused by semilunar valves -- Aortic ( A ) and Pulmonary / Pulmonic ( P ) .
Diagram showing relations of opened heart to front of thoracic wall . Ant . Anterior segment of tricuspid valve . A O. Aorta . A.P. Anterior papillary muscle .
In . Innominate artery . L.C.C. Left common carotid artery . L.S. Left subclavian artery . L.V. Left ventricle . P.A. Pulmonary artery .
</item 2>

<item 3>
Heart sounds
L.V. Left ventricle . P.A. Pulmonary artery . R.A. Right atrium . R.V. Right ventricle . V.S. Ventricular septum . Heart sounds are the noises generated by the
beating heart and the resultant flow of blood through it . Specifically , the sounds reflect the turbulence created when the heart valves snap shut . In cardiac
auscultation , an examiner may use a stethoscope to listen for these unique and distinct sounds that provide important auditory data regarding the condition of
the heart . In healthy adults , there are two normal heart sounds , often described as a lub and a dub ( or dup ) , that occur in sequence with each heartbeat .
These are the first heart sound ( S ) and second heart sound ( S ) , produced by the closing of the atrioventricular valves and semilunar valves , respectively
. In addition to these normal sounds , a variety of other sounds may be present including heart murmurs , adventitious sounds , and gallop rhythms S and S .
Heart murmurs are generated by turbulent flow of blood , which may occur inside or outside the heart .
</item 3>

<item 4>
Heart sounds arise mainly from turbulent blood flow and valve closure, assessed by auscultation. S1 (“lub”) marks mitral and tricuspid closure at systole’s
start; S2 (“dub”) marks aortic and pulmonary closure at systole’s end. S2 normally splits with inhalation. S3 may be normal in youth but later suggest heart
failure or volume overload; S4 usually indicates a stiff ventricle. Murmurs reflect turbulent flow and may be benign
</item 4>

<item 5>
Heart sounds
The reason the third heart sound does not occur until the middle third of diastole is probably that during the early part of diastole , the ventricles are not
filled sufficiently to create enough tension for reverberation . It may also be a result of tensing of the chordae tendineae during rapid filling and expansion
of the ventricle . In other words , an S3 heart sound indicates increased volume of blood within the ventricle . An S3 heart sound is best heard with the bell -
side of the stethoscope ( used for lower frequency sounds ) . A left - sided S3 is best heard in the left lateral decubitus position and at the apex of the
heart , which is normally located in the 5th left intercostal space at the midclavicular line . A right - sided S3 is best heard at the lower - left sternal
border . The way to distinguish between a left and right - sided S3 is to observe whether it increases in intensity with inhalation or exhalation . A right -
sided S3 will increase on inhalation , while a left - sided S3 will increase on exhalation .
</item 5>

<item 6>
Heart sounds
See also ( edit ) Pulse Physical examination Precordial examination Benign pediatric heart murmur Iambic pentameter , a metre in poetry which follows a similar
rhythm to the human heartbeat Pulsatile tinnitus - hearing a heartbeat sound in one or both ears Notes and references ( edit ) Jump up ^
https://depts.washington.edu/physdx/heart/tech2.html Jump up ^ https://depts.washington.edu/physdx/heart/tech2.html Jump up ^ `` thrill '' . Medline Plus
Medical Dictionary . Jump up ^ Bernstein , Lenny ( 2016 - 01 - 02 ) , `` Heart doctors are listening for clues to the future of their stethoscopes '' ,
Washington Post . External links ( edit ) University of Michigan Heart Sound and Murmur Library . Auscultation Assistant . UCLA Heart Sounds - Heart Murmurs .
practicalclinicalskills.com Physiology of the cardiovascular system Heart Cardiac output Cardiac cycle Cardiac output Heart rate Stroke volume Stroke volume End
- diastolic volume End - systolic volume Afterload Preload Frank -- Starling law Cardiac function curve Venous return curve Wiggers diagram Pressure volume
diagram Ultrasound Fractional
</item 6>

<item 7>
Heart sounds
Aortic valve ( to aorta ) right second intercostal space upper right sternal border Pulmonary valve ( to pulmonary trunk ) left second intercostal space upper
left sternal border Erb 's point Left third intercostal space left sternal border Tricuspid valve ( to right ventricle ) left fourth , fifth intercostal spaces
lower left sternal border Mitral valve ( to left ventricle ) left fifth intercostal space left midclavicular line Recording heart sounds ( edit ) Using
electronic stethoscopes , it is possible to record heart sounds via direct output to an external recording device , such as a laptop or MP3 recorder . The same
connection can be used to listen to the previously recorded auscultation through the stethoscope headphones , allowing for more detailed study of murmurs and
other heart sounds , for general research as well as evaluation of a particular patient 's condition .
</item 7>

<item 8>
Heart sounds
The sudden tensing of the chordae tendineae and the squeezing of the ventricles against closed semilunar valves , sends blood rushing back toward the atria ,
and the parachute - like valves catch the rush of blood in their leaflets causing the valve to snap shut . The S1 sound results from reverberation within the
blood associated with the sudden block of flow reversal by the valves . The delay of T1 even more than normally causes the split S1 which is heard in a right
bundle branch blockage . The second heart sound , or S , forms the `` dub '' of `` lub - dub '' and is composed of components A ( aortic valve closure ) and P (
pulmonary valve closure ) . Normally A precedes P especially during inspiration where a split of S can be heard . It is caused by the closure of the semilunar
valves ( the aortic valve and pulmonary valve ) at the end of ventricular systole and the beginning of ventricular diastole . As the left ventricle empties ,
its pressure falls below the pressure in the aorta .
</item 8>

<item 9>
Heart sounds
Main article : Third heart sound Rarely , there may be a third heart sound also called a protodiastolic gallop , ventricular gallop , or informally the ``
Kentucky '' gallop as an onomatopoeic reference to the rhythm and stress of S1 followed by S2 and S3 together ( S1 = Ken ; S2 = tuck ; S3 = y ) . `` lub - dub -
ta '' or `` slosh - ing - in '' If new , indicates heart failure or volume overload . It occurs at the beginning of diastole after S2 and is lower in pitch than
S1 or S2 as it is not of valvular origin . The third heart sound is benign in youth , some trained athletes , and sometimes in pregnancy but if it re-emerges
later in life it may signal cardiac problems , such as a failing left ventricle as in dilated congestive heart failure ( CHF ) . S3 is thought to be caused by
the oscillation of blood back and forth between the walls of the ventricles initiated by blood rushing in from the atria .
</item 9>

<item 10>
Heart sounds
This is a characteristic scratching , creaking , high - pitched sound emanating from the rubbing of both layers of inflamed pericardium . It is the loudest in
systole , but can often be heard at the beginning and at the end of diastole . It is very dependent on body position and breathing , and changes from hour to
hour . Surface anatomy ( edit ) The aortic area , pulmonic area , tricuspid area and mitral area are areas on the surface of the chest where the heart is
auscultated . Heart sounds result from reverberation within the blood associated with the sudden block of flow reversal by the valves closing . Because of this
, auscultation to determine function of a valve is usually not performed at the position of the valve , but at the position to where the sound waves reverberate
.
</item 10>
=== END ITEM id=e7b256f5a398e61c ===

=== ITEM id=c93173eac016f9df ===
Gold sentences and facts:
Sentence 0: A primary ossification center is the first area of a bone to start ossifying . It usually appears during prenatal development in the central part of each developing bone . In long bones the primary centers occur in the diaphysis / shaft and in irregular bones the primary centers occur usually in the body of the bone . Most bones have only one primary center ( e.g. all long bones ) but some irregular bones such as the os coxa ( hip ) and vertebrae have multiple primary centers .
  Fact 0.0: A primary ossification center is the first area of a bone to start ossifying.
  Fact 0.1: A primary ossification center usually appears during prenatal development.
  Fact 0.2: A primary ossification center usually appears in the central part of each developing bone.
  Fact 0.3: In long bones, the primary centers occur in the diaphysis (shaft).
  Fact 0.4: In irregular bones, the primary centers usually occur in the body of the bone.
  Fact 0.5: Most bones have only one primary center, such as all long bones.
  Fact 0.6: Some irregular bones have multiple primary centers.
  Fact 0.7: The os coxa (hip) is an irregular bone with multiple primary centers.
  Fact 0.8: Vertebrae have multiple primary centers.

Retrieved items:
<item 1>
Brain
Once neurons have positioned themselves , their axons sprout and navigate through the brain , branching and extending as they go , until the tips reach their
targets and form synaptic connections . In a number of parts of the nervous system , neurons and synapses are produced in excessive numbers during the early
stages , and then the unneeded ones are pruned away . For vertebrates , the early stages of neural development are similar across all species . As the embryo
transforms from a round blob of cells into a wormlike structure , a narrow strip of ectoderm running along the midline of the back is induced to become the
neural plate , the precursor of the nervous system . The neural plate folds inward to form the neural groove , and then the lips that line the groove merge to
enclose the neural tube , a hollow cord of cells with a fluid - filled ventricle at the center . At the front end , the ventricles and cord swell to form three
vesicles that are the precursors of the forebrain , midbrain , and hindbrain .
</item 1>

<item 2>
Middle cranial fossa
p. 154 . ISBN 9781437735802 . Additional images ( edit ) Middle cranial fossa at human foetus Base of skull Middle cranial fossa Middle cranial fossa Middle
cranial fossa External links ( edit ) Wikimedia Commons has media related to Middle cranial fossa . Anatomy photo : 22 : os - 0802 at the SUNY Downstate Medical
Center Compound structures of skull Neurocranium Calvaria Diploë Asterion Pterion Stephanion Inion Bregma Lambda Fossae anterior middle posterior cranial cavity
Base of skull Fontanelle anterior posterior sphenoidal mastoid Facial skeleton Nasion Gonion Both dacryon zygomatic arch temporal fossa infratemporal fossa
pterygomaxillary fissure pterygopalatine fossa Foramina of the skull ( and canals , fissures , meatus , and hiatus ) Anterior cranial fossa to Orbit : ethmoidal
foramina anterior posterior to Nasal cavity : olfactory foramina ( CN - I ) foramen cecum Middle cranial fossa to Orbit : optic canal ( CN - II ) superior
orbital fissure ( CN - III ,
</item 2>

<item 3>
Ossification center
Ossification center - wikipedia Ossification center Jump to : navigation , search This article needs additional citations for verification . Please help improve
this article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( May 2011 ) ( Learn how and when to remove this
template message ) Ossification center X-ray of ossification centers in a child . Details Identifiers Latin centrum ossificationis Dorlands / Elsevier 12227335
TA A02. 0.00. 043 FMA 75436 Anatomical terminology ( edit on Wikidata ) The first step in ossification of the cartilage is that the cartilage cells , at the
point where ossification is commencing and which is termed as an ossification center , enlarge and arrange themselves in rows . The matrix in which they are
imbedded increases in quantity , so that the cells become further separated from each other . A deposit of calcareous material now takes place in this matrix ,
between the rows of cells , so that they become separated from each other by longitudinal columns of calcified matrix , presenting a granular and opaque
appearance .
</item 3>

<item 4>
Ossification center
Most bones have only one primary center ( e.g. all long bones ) but some irregular bones such as the os coxa ( hip ) and vertebrae have multiple primary centers
. A secondary ossification center is the area of ossification that appears after the primary ossification center has already appeared -- most of which appear
during the postnatal and adolescent years . Most bones have more than one secondary ossification center . In long bones , the secondary centres appear in the
epiphyses . References ( edit ) This article incorporates text in the public domain from the 20th edition of Gray 's Anatomy ( 1918 ) Jump up ^ Gray and Spitzka
( 1910 ) , page 44 . Bibliography ( edit ) Gray , Henry ; Spitzka , Edward Anthony ( 1910 ) . Anatomy , descriptive and applied . the University of California :
Lea & Febiger .
</item 4>

<item 5>
Retina
The role of Pax6 in eye development was elegantly demonstrated by Walter Gehring and colleagues , who showed that ectopic expression of Pax6 can lead to eye
formation on Drosophila antennae , wings , and legs . The optic vesicle gives rise to three structures : the neural retina , the retinal pigmented epithelium ,
and the optic stalk . The neural retina contains the retinal progenitor cells ( RPCs ) that give rise to the seven cell types of the retina . Differentiation
begins with the retinal ganglion cells and concludes with production of the Muller glia . Although each cell type differentiates from the RPCs in a sequential
order , there is considerable overlap in the timing of when individual cell types differentiate . The cues that determine a RPC daughter cell fate are coded by
multiple transcription factor families including the bHLH and homeodomain factors . In addition to guiding cell fate determination , cues exist in the retina to
determine the dorsal - ventral ( D-V ) and nasal - temporal ( N-T ) axes .
</item 5>

<item 6>
Human fertilization
1 Cortical reaction 2 Fusion 2.1 Cell membranes 2.2 Transformations 2.3 Replication 2.4 Mitosis 3 Fertilization age 4 Diseases 5 See also 6 References 7
External links Anatomy ( edit ) Corona radiata ( edit ) The sperm binds through the corona radiata , a layer of follicle cells on the outside of the secondary
oocyte . Fertilization occurs when the nucleus of both a sperm and an egg fuse to form a diploid cell , known as zygote . The successful fusion of gametes forms
a new organism . Cone of attraction and perivitelline membrane ( edit ) Where the spermatozoon is about to pierce , the yolk ( ooplasm ) is drawn out into a
conical elevation , termed the cone of attraction or reception cone . Once the spermatozoon has entered , the peripheral portion of the yolk changes into a
membrane , the perivitelline membrane , which prevents the passage of additional spermatozoa .
</item 6>

<item 7>
Ossification center
Here and there the matrix between two cells of the same row also becomes calcified , and transverse bars of calcified substance stretch across from one
calcareous column to another . Thus there are longitudinal groups of the cartilage cells enclosed in oblong cavities , the walls of which are formed of
calcified matrix which cuts off all nutrition from the cells ; the cells , in consequence , atrophy , leaving spaces called the primary areolæ . Types of
ossification centers ( edit ) Section of fetal bone of cat . ir . Irruption of the subperiosteal tissue . p . Fibrous layer of the periosteum . o . Layer of
osteoblasts . im . Subperiosteal bony deposit . There are two types of ossification centers -- primary and secondary . A primary ossification center is the
first area of a bone to start ossifying . It usually appears during prenatal development in the central part of each developing bone . In long bones the primary
centers occur in the diaphysis / shaft and in irregular bones the primary centers occur usually in the body of the bone .
</item 7>

<item 8>
Ossification center
Anatomy , descriptive and applied . the University of California : Lea & Febiger . Development of bone Ossification Triradiate cartilage Limb development Limb
bud Apical ectodermal ridge Zone of polarizing activity Sclerotome Myotome Septum transversum Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Ossification_center&oldid=808049222 '' Categories : Wikipedia articles incorporating text from the 20th edition of
Gray 's Anatomy ( 1918 ) Skeletal system Tissues ( biology ) Hidden categories : Articles needing additional references from May 2011 All articles needing
additional references Medicine infobox template using GraySubject or GrayPage Medicine infobox template using Dorlands parameter Talk About Wikipedia Nederlands
Edit links This page was last edited on 31 October 2017 , at 15 : 42 . About Wikipedia
</item 8>

<item 9>
Brain
At the next stage , the forebrain splits into two vesicles called the telencephalon ( which will contain the cerebral cortex , basal ganglia , and related
structures ) and the diencephalon ( which will contain the thalamus and hypothalamus ) . At about the same time , the hindbrain splits into the metencephalon (
which will contain the cerebellum and pons ) and the myelencephalon ( which will contain the medulla oblongata ) . Each of these areas contains proliferative
zones where neurons and glial cells are generated ; the resulting cells then migrate , sometimes for long distances , to their final positions . Once a neuron
is in place , it extends dendrites and an axon into the area around it . Axons , because they commonly extend a great distance from the cell body and need to
reach specific targets , grow in a particularly complex way . The tip of a growing axon consists of a blob of protoplasm called a growth cone , studded with
chemical receptors .
</item 9>

<item 10>
Gallbladder
Early in development , the human embryo has three germ layers and abuts an embryonic yolk sac . During the second week of embryogenesis , as the embryo grows ,
it begins to surround and envelop portions of this sac . The enveloped portions form the basis for the adult gastrointestinal tract . Sections of this foregut
begin to differentiate into the organs of the gastrointestinal tract , such as the esophagus , stomach , and intestines . During the fourth week of
embryological development , the stomach rotates . The stomach , originally lying in the midline of the embryo , rotates so that its body is on the left . This
rotation also affects the part of the gastrointestinal tube immediately below the stomach , which will go on to become the duodenum . By the end of the fourth
week , the developing duodenum begins to spout a small outpouching on its right side , the hepatic diverticulum , which will go on to become the biliary tree .
Just below this is a second outpouching , known as the cystic diverticulum , that will eventually develop into the gallbladder . Function ( edit ) 1 .
</item 10>
=== END ITEM id=c93173eac016f9df ===

=== ITEM id=773bc149e09d237c ===
Gold sentences and facts:
Sentence 0: The Earth 's magnetic field serves to deflect most of the solar wind , whose charged particles would otherwise strip away the ozone layer that protects the Earth from harmful ultraviolet radiation . One stripping mechanism is for gas to be caught in bubbles of magnetic field , which are ripped off by solar winds . Calculations of the loss of carbon dioxide from the atmosphere of Mars , resulting from scavenging of ions by the solar wind , indicate that the dissipation of the magnetic field of Mars caused a near total loss of its atmosphere .
  Fact 0.0: The Earth's magnetic field serves to deflect most of the solar wind.
  Fact 0.1: The solar wind's charged particles would otherwise strip away the ozone layer.
  Fact 0.2: The ozone layer protects the Earth from harmful ultraviolet radiation.
  Fact 0.3: One stripping mechanism is for gas to be caught in bubbles of magnetic field.
  Fact 0.4: The bubbles of magnetic field are ripped off by solar winds.
  Fact 0.5: Calculations of the loss of carbon dioxide from the atmosphere of Mars result from scavenging of ions by the solar wind.
  Fact 0.6: The calculations indicate that the dissipation of the magnetic field of Mars caused a near total loss of the atmosphere of Mars.

Retrieved items:
<item 1>
Earth's magnetic field
Magnetosphere ( edit ) An artist 's rendering of the structure of a magnetosphere. 1 ) Bow shock . 2 ) Magnetosheath. 3 ) Magnetopause. 4 ) Magnetosphere. 5 )
Northern tail lobe. 6 ) Southern tail lobe. 7 ) Plasmasphere . Main article : Magnetosphere Earth 's magnetic field , predominantly dipolar at its surface , is
distorted further out by the solar wind . This is a stream of charged particles leaving the Sun 's corona and accelerating to a speed of 200 to 1000 kilometres
per second . They carry with them a magnetic field , the interplanetary magnetic field ( IMF ) . The solar wind exerts a pressure , and if it could reach Earth
's atmosphere it would erode it . However , it is kept away by the pressure of the Earth 's magnetic field . The magnetopause , the area where the pressures
balance , is the boundary of the magnetosphere .
</item 1>

<item 2>
Earth's magnetic field
Earth 's magnetic field - wikipedia Earth 's magnetic field Jump to : navigation , search Computer simulation of the Earth 's field in a period of normal
polarity between reversals . The lines represent magnetic field lines , blue when the field points towards the center and yellow when away . The rotation axis
of the Earth is centered and vertical . The dense clusters of lines are within the Earth 's core . Earth 's magnetic field , is also known as the geomagnetic
field , is the magnetic field that extends from the Earth 's interior out into space , where it meets the solar wind , a stream of charged particles emanating
from the Sun . Its magnitude at the Earth 's surface ranges from 25 to 65 microteslas ( 0.25 to 0.65 gauss ) . Roughly speaking it is the field of a magnetic
dipole currently tilted at an angle of about 11 degrees with respect to Earth 's rotational axis , as if there were a bar magnet placed at that angle at the
center of the Earth .
</item 2>

<item 3>
Earth's magnetic field
It extends several tens of thousands of kilometers into space , protecting the Earth from the charged particles of the solar wind and cosmic rays that would
otherwise strip away the upper atmosphere , including the ozone layer that protects the Earth from harmful ultraviolet radiation . Contents ( hide ) 1
Importance 2 Main characteristics 2.1 Description 2.1. 1 Intensity 2.1. 2 Inclination 2.1. 3 Declination 2.2 Geographical variation 2.3 Dipolar approximation
2.4 Magnetic poles 3 Magnetosphere 4 Time dependence 4.1 Short - term variations 4.2 Secular variation 4.3 Magnetic field reversals 4.4 Earliest appearance 4.5
Future 5 Physical origin 5.1 Earth 's core and the geodynamo 5.1. 1 Numerical models 5.2 Currents in the ionosphere and magnetosphere 6 Measurement and analysis
6.1 Detection 6.2 Crustal magnetic anomalies 6.3 Statistical models 6.3. 1 Spherical harmonics 6.3. 1.1 Radial dependence 6.3.
</item 3>

<item 4>
Earth's magnetic field
Our Magnetic Earth : The Science of Geomagnetism . University of Chicago Press . ISBN 0 - 226 - 52050 - 1 . Merrill , Ronald T. ; McElhinny , Michael W. ;
McFadden , Phillip L. ( 1996 ) . The magnetic field of the earth : paleomagnetism , the core , and the deep mantle . Academic Press . ISBN 978 - 0 - 12 - 491246
- 5 . `` Temperature of the Earth 's core '' . NEWTON Ask a Scientist. 1999 . Tauxe , Lisa ( 1998 ) . Paleomagnetic Principles and Practice . Kluwer . ISBN 0 -
7923 - 5258 - 0 . Towle , J.N. ( 1984 ) . `` The Anomalous Geomagnetic Variation Field and Geoelectric Structure Associated with the Mesa Butte Fault System ,
Arizona '' . Geological Society of America Bulletin . 9 ( 2 ) : 221 -- 225 .
</item 4>

<item 5>
Earth's magnetic field
However the process is complex , and computer models that reproduce some of its features have only been developed in the last few decades . Earth 's core and
the geodynamo ( edit ) A schematic illustrating the relationship between motion of conducting fluid , organized into rolls by the Coriolis force , and the
magnetic field the motion generates . The Earth and most of the planets in the Solar System , as well as the Sun and other stars , all generate magnetic fields
through the motion of electrically conducting fluids . The Earth 's field originates in its core . This is a region of iron alloys extending to about 3400 km (
the radius of the Earth is 6370 km ) . It is divided into a solid inner core , with a radius of 1220 km , and a liquid outer core . The motion of the liquid in
the outer core is driven by heat flow from the inner core , which is about 6,000 K ( 5,730 ° C ; 10,340 ° F ) , to the core - mantle boundary , which is about
3,800 K ( 3,530 ° C ; 6,380 ° F ) .
</item 5>

<item 6>
Earth's magnetic field
The pattern of flow is organized by the rotation of the Earth and the presence of the solid inner core . The mechanism by which the Earth generates a magnetic
field is known as a dynamo . The magnetic field is generated by a feedback loop : current loops generate magnetic fields ( Ampère 's circuital law ) ; a
changing magnetic field generates an electric field ( Faraday 's law ) ; and the electric and magnetic fields exert a force on the charges that are flowing in
currents ( the Lorentz force ) .
</item 6>

<item 7>
Earth's magnetic field
( 2008 ) . Discovering the Essential Universe ( Fourth ed . ) . W.H. Freeman . ISBN 978 - 1 - 4292 - 1797 - 2 . Herndon , J.M. ( 1996 - 01 - 23 ) . ``
Substructure of the inner core of the Earth '' . PNAS. 93 ( 2 ) : 646 -- 648 . Bibcode : 1996PNAS ... 93 ... 646H . doi : 10.1073 / pnas. 93.2. 646 . PMC 40105
. PMID 11607625 . Hollenbach , D.F. ; Herndon , J.M. ( 2001 - 09 - 25 ) . `` Deep - Earth reactor : Nuclear fission , helium , and the geomagnetic field '' .
PNAS. 98 ( 20 ) : 11085 -- 90 . Bibcode : 2001PNAS ... 9811085H . doi : 10.1073 / pnas. 201393998 .
</item 7>

<item 8>
Earth's magnetic field
1.1 Radial dependence 6.3. 2 Global models 7 Biomagnetism 8 See also 9 References 10 Further reading 11 External links Importance ( edit ) The Earth 's magnetic
field serves to deflect most of the solar wind , whose charged particles would otherwise strip away the ozone layer that protects the Earth from harmful
ultraviolet radiation . One stripping mechanism is for gas to be caught in bubbles of magnetic field , which are ripped off by solar winds . Calculations of the
loss of carbon dioxide from the atmosphere of Mars , resulting from scavenging of ions by the solar wind , indicate that the dissipation of the magnetic field
of Mars caused a near total loss of its atmosphere . The study of past magnetic field of the Earth is known as paleomagnetism . The polarity of the Earth 's
magnetic field is recorded in igneous rocks , and reversals of the field are thus detectable as `` stripes '' centered on mid-ocean ridges where the sea floor
is spreading , while the stability of the geomagnetic poles between reversals has allowed paleomagnetists to track the past motion of continents .
</item 8>

<item 9>
Earth's magnetic field
Currents in the ionosphere and magnetosphere ( edit ) Electric currents induced in the ionosphere generate magnetic fields ( ionospheric dynamo region ) . Such
a field is always generated near where the atmosphere is closest to the Sun , causing daily alterations that can deflect surface magnetic fields by as much as
one degree . Typical daily variations of field strength are about 25 nanoteslas ( nT ) ( one part in 2000 ) , with variations over a few seconds of typically
around 1 nT ( one part in 50,000 ) . Measurement and analysis ( edit ) Detection ( edit ) The Earth 's magnetic field strength was measured by Carl Friedrich
Gauss in 1832 and has been repeatedly measured since then , showing a relative decay of about 10 % over the last 150 years . The Magsat satellite and later
satellites have used 3 - axis vector magnetometers to probe the 3 - D structure of the Earth 's magnetic field . The later Ørsted satellite allowed a comparison
indicating a dynamic geodynamo in action that appears to be giving rise to an alternate pole under the Atlantic Ocean west of S. Africa .
</item 9>

<item 10>
Earth's magnetic field
Reversals also provide the basis for magnetostratigraphy , a way of dating rocks and sediments . The field also magnetizes the crust , and magnetic anomalies
can be used to search for deposits of metal ores . Humans have used compasses for direction finding since the 11th century A.D. and for navigation since the
12th century . Although the magnetic declination does shift with time , this wandering is slow enough that a simple compass remains useful for navigation .
Using magnetoception various other organisms , ranging from some types of bacteria to pigeons , use the Earth 's magnetic field for orientation and navigation .
Main characteristics ( edit ) Description ( edit ) At any location , the Earth 's magnetic field can be represented by a three - dimensional vector . A typical
procedure for measuring its direction is to use a compass to determine the direction of magnetic North . Its angle relative to true North is the declination ( D
) or variation . Facing magnetic North , the angle the field makes with the horizontal is the inclination ( I ) or magnetic dip . The intensity ( F ) of the
field is proportional to the force it exerts on a magnet .
</item 10>
=== END ITEM id=773bc149e09d237c ===
