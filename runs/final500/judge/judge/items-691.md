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

=== ITEM id=681bf60dafbdbdf7 ===
Gold sentences and facts:
Sentence 0: As a young woman , Rogue permanently absorbed the superhuman powers and the psyche of Carol Danvers , the original Ms. Marvel . This provided her with superhuman strength , stamina , durability , reflexes , speed and a 7th sense . She was able to repel bullets and fly at sub-sonic speeds , much like Ms. Marvel could . In addition , she possessed an amalgamated mutant human / Kree physiology that rendered her resistant to most toxins and poisons , with the added effect of making her virtually invulnerable .
  Fact 0.0: Rogue, as a young woman, permanently absorbed the superhuman powers of Carol Danvers.
  Fact 0.1: Rogue, as a young woman, permanently absorbed the psyche of Carol Danvers.
  Fact 0.2: Carol Danvers is the original Ms. Marvel.
  Fact 0.3: Rogue's absorption gave Rogue superhuman strength.
  Fact 0.4: Rogue's absorption gave Rogue superhuman stamina.
  Fact 0.5: Rogue's absorption gave Rogue superhuman durability.
  Fact 0.6: Rogue's absorption gave Rogue superhuman reflexes.
  Fact 0.7: Rogue's absorption gave Rogue superhuman speed.
  Fact 0.8: Rogue's absorption gave Rogue a 7th sense.
  Fact 0.9: Rogue was able to repel bullets.
  Fact 0.10: Rogue was able to fly at sub-sonic speeds.
  Fact 0.11: Rogue's ability to fly at sub-sonic speeds was much like Ms. Marvel's.
  Fact 0.12: Rogue possessed an amalgamated mutant human/Kree physiology.
  Fact 0.13: Rogue's amalgamated mutant human/Kree physiology made Rogue resistant to most toxins.
  Fact 0.14: Rogue's amalgamated mutant human/Kree physiology made Rogue resistant to most poisons.
  Fact 0.15: The added effect of Rogue's amalgamated physiology was that Rogue was virtually invulnerable.

Retrieved items:
<item 1>
Rogue (comics)
Over time , Rogue 's own abilities returned , although exactly how and when was never revealed or discussed . However , she no longer possessed the abilities
she had absorbed from Ms. Marvel . Rogue and Gambit were both put on Havok 's team and participated in various missions . As described in her own miniseries ,
Rogue then traveled down south to save a young mutant girl from her powers . While there , she met Campbell St. Ange , a young man who was immune to Rogue 's
lethal touch . Also while there , Rogue forcibly absorbed knowledge from her Aunt Carrie that explained that Rogue 's mother had traveled to the Far Banks , a
dream - realm , to stop her father from getting there himself . Rogue encountered the incorporeal spirit of her mother therein , and absorbed her memories .
After the reunion , her mother 's trapped spirit could finally move on . Rogue subsequently went back to her Aunt Carrie and made amends with her .
</item 1>

<item 2>
Rogue (comics)
She possesses superhuman strength sufficient enough to lift well over 100 tons , as well as some degree of superhuman speed , agility , and reflexes . In
addition , Rogue is virtually invulnerable to damage and is capable of flying at high speeds . Rogue 's eyes glow in darkness and her vision extends somewhat
into the infrared spectrum , allowing her to see in the dark . Her new ionic physiology also grants her a form of immortality , protecting her from age and
disease . Natural skills ( edit ) Rogue has shown a fair knowledge of unarmed combat when in a position where she has to fight in situations where she can not
simply use her powers to knock her foe out . She has also received a degree of training from Professor X to protect herself against even subtle telepathic
attacks , this training allowing her to escape Kobik 's reprogramming of various Avengers and villains to reassemble her teammates against this new threat when
even experienced telepaths like Cable were trapped in the illusion . Rogue 's real name ( edit ) In the Going Rogue arc of her 2004 ongoing series , Rogue 's
real name is finally established as Anna Marie .
</item 2>

<item 3>
Rogue (comics)
Ms. Marvel powers ( edit ) As a young woman , Rogue permanently absorbed the superhuman powers and the psyche of Carol Danvers , the original Ms. Marvel . This
provided her with superhuman strength , stamina , durability , reflexes , speed and a 7th sense . She was able to repel bullets and fly at sub-sonic speeds ,
much like Ms. Marvel could . In addition , she possessed an amalgamated mutant human / Kree physiology that rendered her resistant to most toxins and poisons ,
with the added effect of making her virtually invulnerable . Rogue on the cover of Rogue vol 3 # 7 ( March 2005 ) ; art by Frank G. D'Armata , Scot Eaton , and
Don Hillsman Rogue also gained a precognitive `` seventh sense '' that enables her to predict an enemy 's move subconsciously during battle . She used this
ability to predict where Nightcrawler would teleport and from which direction Magus of the Technarchy would attack . This ability was not always reliable ,
however , and would randomly and sporadically activate .
</item 3>

<item 4>
Rogue (comics)
When Professor Xavier removes the blocks , her powers are allowed to develop normally . With this new control , Rogue demonstrates the ability to absorb and
collectively utilize all the powers of the New X-Men , in order to defeat a rogue Predator X , with no apparent harm to either them or herself . Her power also
affects aliens , such as when she was transferred to a planet in another dimension by Magik , and one of the aliens involuntarily touches her and she gains the
ability to communicate in their language , along with their knowledge . She later uses a more lethal version of her powers against the Avengers during her fight
with them , even making Falcon and She - Hulk immediately unconscious without any negative feedback on her physical self ( besides She - Hulk 's green skin )
but she can not control the minds of the ones she absorbs . She absorbs their powers and renders them unconscious for a long time . The new twist in Rogue 's
powers consists of prompting her victims of her skin - to - skin contact , for a consent on the process .
</item 4>

<item 5>
Rogue (comics)
Therefore , Rogue considers her powers to be a curse . For most of her life , she limited her physical contact with others , including her on - off love
interest , Gambit . However , after many years , Rogue finally gained full control over her mutant ability . Hailing from the fictional Caldecott County ,
Mississippi , Rogue is the X-Men's self - described southern belle . A runaway , she was adopted by Mystique of the Brotherhood of Evil Mutants and grew up as a
villain . After Rogue permanently absorbs Ms. Marvel 's psyche and Kree powers , she reforms and turns to the X-Men , fearing for her sanity . Rogue 's real
name and early history were not revealed until nearly 20 years after her introduction . Until the back story provided by Robert Rodi in the ongoing Rogue series
, which began in September 2004 , Rogue 's background was only hinted at . Her name was revealed as Anna Marie , although her surname is still unknown . She has
sometimes adopted the name Raven , which is the first name of her foster mother , Mystique .
</item 5>

<item 6>
Rogue (comics)
She also still possessed a `` fluid genome '' that enabled Sage to use her as a conduit through which to channel the mutant powers of the X-Treme X-Men team in
a fight against Bogan . Sunfire powers ( edit ) In the short - lived volume three of her self - entitield series ( 2004 -- 2005 ) , Rogue absorbed a large
portion of the mutant Sunfire 's solar - absorption based powers . In addition to her own natural mutant abilities Rogue could then project intense heat and
flame , envelop her body in a fiery aura , fly by focusing her power downwards in a tight stream to propel her like a rocket , focus her power inward to
increase her strength ( though not at her Ms. Marvel levels ) , exercise immunity to heat and radiation , and see the infra - red spectrum . Rogue was
eventually purged of these abilities when she came into contact with Hope Summers . Wonder Man powers ( edit ) Rogue currently possesses Wonder Man 's essence ,
including his powers and psyche . Her organic tissues have been permeated with ionic energy , granting her an array of superhuman abilities .
</item 6>

<item 7>
Rogue (comics)
Powers and abilities ( edit ) This section relies too much on references to primary sources . Please improve this section by adding secondary or tertiary
sources . ( March 2015 ) ( Learn how and when to remove this template message ) This section describes a work or element of fiction in a primarily in - universe
style . Please help rewrite it to explain the fiction more clearly and provide non-fictional perspective . ( March 2015 ) ( Learn how and when to remove this
template message ) Due to the nature of her unique mutant powers , Rogue has had several different abilities over the decades . Mutant powers ( edit ) Cover art
of Rogue vol. 3 , # 3 ( September 2004 ) ; art by Rodolfo Migliari . Rogue possesses the mutant ability to absorb the psyche and abilities of another human
being ( or members of some sapient alien races ) through skin contact . Rogue can absorb the memories , knowledge , talents , personality and physical abilities
( whether superhuman or not ) of the person she touches , as well as occasionally duplicating in herself physical characteristics of her victim .
</item 7>

<item 8>
Rogue (comics)
Through meditation , Rogue learned how to suppress the random manifestations of most of these abilities , but found herself with a new ' default ' state for a
time , where she possessed Wolverine 's claws and healing factor , Cyclops ' optic blasts - requiring her to wear ruby quartz glasses - while also still
retaining her mutant and Ms. Marvel powers . Later , Sage used her own mutant abilities to ' jump - start ' Rogue 's power , giving her the control to manifest
any ability she had ever absorbed , whenever she wished to . It is generally accepted that after regaining her abilities after X-Treme X-Men , she has lost this
ability . Though Z'Cann was a telepath , Rogue never displayed this power ( though she did manifest telepathy , it was with the Phoenix energy form , so it was
most likely an echo of Jean Grey 's powers ) . Powerless ( edit ) For a time Rogue lost all of her powers , including her original mutant ones , after she was
fatally injured in a fight with the mysterious warrior Vargas . During this period , she displayed exceptional fighting skills and agility , though it was said
these were not superhuman in nature .
</item 8>

<item 9>
Graph query row:
entity: Rogue
relationship: RELATED_TO
related: Carol Danvers
fact: Rogue permanently absorbed Carol Danvers's powers and psyche.
context: None
detail: None
relation: bumps into
role: None
</item 9>

<item 10>
Rogue (comics)
This ability was not always reliable , however , and would randomly and sporadically activate . When she possessed Carol Danvers ' psyche , her `` double ''
consciousness made her highly resistant to telepathic probes , even those of Charles Xavier , which was said to have been a byproduct of two minds existing in
one body and / or Carol Danvers ' Kree physiology . The Ms. Marvel psyche was eventually torn out of Rogue in a subsequent issue , and it was destroyed by
Magneto . X-Treme abilities ( edit ) During the Maximum Security miniseries , Rogue absorbed an injured Cadre K girl named Z'Cann . The girl held on longer than
necessary , using her mind control to boost the process and because of this ( perhaps because she had absorbed a fluid Skrull physiology ) Rogue mutated further
, gaining the ability to ' recall ' the powers of anyone she had ever absorbed before , though she could not control when or which of these powers surfaced .
For a short time Rogue also had a Skrull appearance , scanned as though she had Skrull DNA , and had their shapeshifting powers , but these soon faded and she
returned to her normal appearance .
</item 10>
=== END ITEM id=681bf60dafbdbdf7 ===

=== ITEM id=733692155846294b ===
Gold sentences and facts:
Sentence 0: In every High Court , there is a Chief Justice and many other judges whose number is defined by the President of India .
  Fact 0.0: Every High Court has a Chief Justice.
  Fact 0.1: Every High Court has many other judges.
  Fact 0.2: The number of other judges in a High Court is defined by the President of India.

Retrieved items:
<item 1>
Justice of the peace
Until the Courts Act 2003 came into force , magistrates were tied to a particular area ( see magistrates ' courts committee , commission area , petty sessions
area ) . This has now been changed such that they are assigned to local justice areas , but less strongly . The Courts Act provides the current framework for
appointment of the justices , which is done by the Lord Chancellor in the name of sovereign . Justices can also be removed by the same mechanism . District
judges ( magistrates ' court ) -- previously known as stipendiary magistrates -- must have a seven years ' general legal qualification , and are appointed by
the sovereign on the advice of the Lord Chancellor . Before 1714 , magistrates were liable to be approached at any time and in any place by people legally
recognised as paupers , appealing for aid if parish authorities refused to provide any . It was relatively common for these magistrates to write out , on the
spot , an order requiring aid to be granted . Scotland ( edit ) See also : Judiciary of Scotland Within the Scottish legal system justices of the peace are lay
magistrates who currently sit in the justice of the peace courts .
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
182 -- Sirisena Sri Lanka Law College University of Exeter Judge of the Court of Appeal Kankanithanthri Chitrasiri † 3 December 2015 -- present --
7002875000000000000 ♠ 875 -- Sirisena University of Sri Lanka Queen Mary University of London Judge of the Court of Appeal Nalin Perera † 3 March 2016 --
present -- 7002784000000000000 ♠ 784 -- Sirisena Judge of the Court of Appeal Prasanna Sujeewa Jayawardena † 16 June 2016 -- present -- 7002679000000000000 ♠
679 -- Sirisena University of Colombo Judge of the Court of Appeal See also ( edit ) List of Chief Justices of Sri Lanka List of Justices of the Supreme Court
of Sri Lanka by court composition Notes ( edit ) Jump up ^ Provisional Chief Justice Jump up ^ On leave ( 1 January - 28 April 1871 ) Jump up ^ Acting Chief
Justice ( 1 - 10 October 1879 ,
</item 3>

<item 4>
Indian High Courts Act 1861
Contents ( hide ) 1 Abolishing Existing Courts 2 Qualifications of High Court Judges 3 Judicial Tenure and Seniority 4 Jurisdiction 5 References Abolishing
existing Courts ( edit ) The Act abolished the Supreme Courts at Calcutta , Madras , and Bombay ; the Sadar Diwani Adalat and the Sadar Nizamat Adalat at
Calcutta ; Sadar Adalat and Faujdari Adalat at Madras ; Sadar Diwani Adalat and Faujdari Adalat at Bombay ( § 8 ) . Qualifications of High Court judges ( edit )
Each High Court could consist of a chief justice and up to 15 judges . Under § 3 of the Act , judges could be selected from barristers ( with 5 years of
experience ) , civil servants ( with 10 years of experience including 3 years as a zillah judge ) , judges of small cause courts or sudder ameen ( with 5 years
of experience ) , or pleaders of lander courts or High Courts ( with 05 years of experience ) .
</item 4>

<item 5>
Indian High Courts Act 1861
It exercises the power of superintendence and control over all courts and tribunals throughout the territory except the military tribunals . Judicial tenure and
seniority ( edit ) Under § 5 of the Act , judges served at the pleasure of Her Majesty . The chief justice had precedence over judges , whereas judges had
seniority based on appointment . Under § 14 , the chief justice had the power to form divisions of two or more judges to hear cases . Jurisdiction ( edit )
Under article 9 of the Act , each High Court had `` all such powers and authority for and in relation to the administration of justice '' including original and
appellate jurisdiction over civil , criminal , admiralty , vice-admiralty , testamentary , intestate , and matrimonial matters . References ( edit ) Jump up ^
`` Indian High Courts Act 1861 '' . GKToday. 2011 - 10 - 17 . Retrieved 2017 - 03 - 20 . Jump up ^ Criminal Justice India Series : Punjab , 2002 . Allied
Publishers . 2002 . p. 233 .
</item 5>

<item 6>
Indian High Courts Act 1861
Appellate Jurisdiction : The appellate jurisdiction of the High Courts extends so : ( a ) The High Court can hear appeals in civil cases if the amount involved
in the case is at least Rs. 5000 . ( b ) The High Court in criminal cases hears the appeal in which the accused has been sentenced to four years imprisonment by
the Sessions Judge . v ( c ) The death sentence awarded by Sessions Judge is subject to approval by the High Court . ( d ) The High Court hear the cases
involving interpretation of the Constitution or Law . ( e ) The High Court hears the cases on income tax , sales tax etc . Power of Judicial Review : The States
High Courts like the Supreme Court has the power of Judicial Review . A High Court has the power to strike down any law of the State or any order of the
executive if it violates any provision of the constitution or curtails or takes any of the Fundamental Rights of the people . Administrative and Supervisory
Power : The State High Court performs many administrative functions within its Territorial Jurisdiction .
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
List of Justices of the Supreme Court of Sri Lanka
Justice of the Straits Settlements 35 George Henry Withers -- -- -- 36 Dodwell Browne -- -- -- 37 Fredrick Charles Moncrieff 4 April 1900 -- -- -- 38 Henry
Lorentz Wendt 15 October 1901 -- 28 November 1910 -- -- 39 John Page Middleton 28 January 1902 -- 1912 -- -- Puisine Judge of the Supreme Court of Cyprus 40
Charles Layard Western 26 April 1902 -- 18 June 1906 26 April 1902 -- 18 June 1906 1,514 1,514 41 Joseph Richard Grenier 28 November 1910 -- 1912 -- -- 42
Alexander Wood Renton 3 September 1905 -- 1918 22 August 1914 - 1918 -- 43 Alfred George Lascelles England 1 May 1911 -- 1914 1 May 1911 -- 1914 -- 44 Joseph
Turner Hutchinson England 23 October 1906 -- 1 May 1911 23 October 1906 -- 1 May 1911 1,
</item 8>

<item 9>
Indian High Courts Act 1861
Salary : The pay of the Chief Justice of a High Court is rupees 90,000 / - per month and that of the other judges is rupees 85,000 / - per month . Powers and
Functions Original Jurisdiction : The original jurisdiction of the High Court is restricted . ( a ) Every High Court under Article 226 is empowered to issue
writs , orders , directions including writs in the nature of Habeas Corpus , Mandamus , Prohibition , Quo - warranto and Certiorari or any of them to any person
or authority with in its territory for the enforcement of the Fundamental Rights and for any other purpose . ( b ) The original jurisdiction of High Court
extends to matters of admiralty , matrimonial , contempt of court and cases ordered to be transferred to High Court by lower court . ( c ) The High Courts of
Mumbai , Kolkata and Chennai have original jurisdiction on hearing straightway cases involving the Christians and Parsies . ( d ) The High Courts of Mumbai ,
Kolkata and Chennai exercise original civil jurisdiction when the amount involved is more than two thousand rupees .
</item 9>

<item 10>
Justice of the peace
Majesty 's Courts and Tribunals Service Judges ' Council Civil and family courts ( show ) Supreme Court of the United Kingdom Privy Council Court of Appeal
Master of the Rolls Court of Appeal judge High Court of Justice President of the Queen 's Bench President of the Family Division Chancellor of the High Court
High Court judge County courts List of county court venues County Court Business Centre District judge Family Court Court of Protection Court of Chivalry
Criminal courts ( show ) Supreme Court of the United Kingdom Court of Appeal Lord Chief Justice Court of Appeal judge High Court of Justice President of the
Queen 's Bench High Court judge Crown Court List of Crown Court venues Circuit judge Recorder Magistrates ' courts District judge Justice of the Peace / Lay
magistrates Criminal prosecution ( show ) Attorney General Director of Public Prosecutions Crown Prosecution Service Legal profession ( show ) Bar Council
Barrister Law Society of England and Wales Solicitor Solicitor Advocate Main article : Magistrate ( England and Wales ) A magistrates ' court in England and
Wales is composed of a bench of ( usually three ) JPs or magistrates , who dispense summary justice : that is they decide on offences which carry up to six
months in prison ,
</item 10>
=== END ITEM id=733692155846294b ===

=== ITEM id=634adc5108050f3a ===
Gold sentences and facts:
Sentence 0: On Earth , Ivan 's Ecto - Morphicons are completely unearthed , and he unleashes them on Angel Grove , ordering the parents to commit suicide at the construction site . Fred , Bulk , Skull and the other kids head to the construction site to save their parents . The Rangers return with their new animal - themed Ninja Zords and destroy one of Ivan 's Ecto - Morphicons . Ivan takes control of the other and battles the Rangers himself . The Rangers lead Ivan into space right into the path of Ryan 's Comet , which destroys him . His destruction breaks the hypnosis and the parents are reunited with their children . The Rangers then use the Great Power to restore the Command Center and resurrect Zordon .
  Fact 0.0: On Earth, Ivan's Ecto-Morphicons are completely unearthed.
  Fact 0.1: Ivan unleashes the Ecto-Morphicons on Angel Grove.
  Fact 0.2: Ivan orders the parents to commit suicide at the construction site.
  Fact 0.3: Fred, Bulk, Skull and the other kids head to the construction site to save their parents.
  Fact 0.4: The Rangers return with their new animal-themed Ninja Zords.
  Fact 0.5: The Rangers destroy one of Ivan's Ecto-Morphicons.
  Fact 0.6: Ivan takes control of the other Ecto-Morphicon.
  Fact 0.7: Ivan battles the Rangers himself.
  Fact 0.8: The Rangers lead Ivan into space, right into the path of Ryan's Comet.
  Fact 0.9: Ryan's Comet destroys Ivan.
  Fact 0.10: Ivan's destruction breaks the hypnosis.
  Fact 0.11: The parents are reunited with their children.
  Fact 0.12: The Rangers use the Great Power to restore the Command Center.
  Fact 0.13: The Rangers use the Great Power to resurrect Zordon.

Retrieved items:
<item 1>
Mighty Morphin Power Rangers: The Movie
For the 2017 film , see Power Rangers ( film ) . Mighty Morphin Power Rangers : The Movie Theatrical release poster Directed by Bryan Spicer Produced by Haim
Saban Shuki Levy Suzanne Todd Screenplay by Arne Olsen Story by John Kamps Arne Olsen Starring Karan Ashley Johnny Yong Bosch Steve Cardenas Jason David Frank
Amy Jo Johnson David Yost Jason Narvy Paul Schrier Paul Freeman Music by Graeme Revell Cinematography Paul Murphy Edited by Wayne Wahrman Production company
Saban Entertainment Toei Company Distributed by 20th Century Fox Release date June 30 , 1995 ( 1995 - 06 - 30 ) Running time 95 minutes Country United States
Language English Budget $15 million Box office $66.4 million Mighty Morphin Power Rangers : The Movie ( also known as Power Rangers : The Movie ) is a 1995
American superhero film based on the television series Mighty Morphin Power Rangers . It stars the ensemble cast of Karan Ashley , Johnny Yong Bosch , Steve
Cardenas , Jason David Frank , Amy Jo Johnson , and David Yost alongside the villains cast from the original series and Paul Freeman as Ivan Ooze .
</item 1>

<item 2>
Mighty Morphin Power Rangers: The Movie
Lord Zedd , Rita Repulsa , Goldar , and Mordant arrive at the construction site and crack open the egg , releasing Ivan Ooze after 6,000 years , a morphological
being who ruled Earth with an iron fist before he was overthrown by Zordon and a group of young warriors . Ivan lays siege to the Rangers ' Command Center and
incapacitates Zordon , robbing the Rangers of their powers . As the Rangers return to the Command Center , they find it destroyed and Zordon dying . Zordon 's
assistant Alpha 5 sends the Rangers to the distant planet Phaedos to obtain the Great Power and save Zordon . On the Moon , Ivan usurps Rita and Zedd , trapping
them in a snow globe , then sends his Tengu warriors to Phaedos and begins building an army . He uses children to bring his ooze to their parents , and it
hypnotizes them into becoming his workforce to dig up his Ecto - Morphicon Titans , twin war machines built during his reign .
</item 2>

<item 3>
Mighty Morphin Power Rangers: The Movie
Fred , Bulk , Skull and the other kids head to the construction site to save their parents . The Rangers return with their new animal - themed Ninja Zords and
destroy one of Ivan 's Ecto - Morphicons . Ivan takes control of the other and battles the Rangers himself . The Rangers lead Ivan into space right into the
path of Ryan 's Comet , which destroys him . His destruction breaks the hypnosis and the parents are reunited with their children . The Rangers then use the
Great Power to restore the Command Center and resurrect Zordon . In a mid-credits scene , Goldar briefly lounges in Zedd 's throne being served by Mordant only
to panic when Zedd and Rita appear having been released after Ivan was destroyed . Cast ( edit ) Karan Ashley as Aisha Campbell , The Yellow Ranger Johnny Yong
Bosch as Adam Park , The Black Ranger Steve Cardenas as Rocky DeSantos , The Red Ranger Jason David Frank as Tommy Oliver , The White Ranger Amy Jo Johnson as
Kimberly Hart , The Pink Ranger David Yost as Billy Cranston ,
</item 3>

<item 4>
Mighty Morphin Power Rangers: The Movie
Force ( 2002 ) Ninja Storm ( 2003 ) Dino Thunder ( 2004 ) S.P.D.
</item 4>

<item 5>
Mighty Morphin Power Rangers: The Movie
The New York Times . Retrieved 2010 - 08 - 07 . Jump up ^ `` Mighty Morphin Power Rangers : The Movie '' . RogerEbert.com . June 30 , 1995 . Retrieved 2010 - 08
- 07 . Jump up ^ Lasalle , Mick ( 1995 - 06 - 30 ) . `` Mighty Mindless ' Rangers ' '' . San Francisco Chronicle . Retrieved 2010 - 08 - 25 . External links (
edit ) Mighty Morphin Power Rangers : The Movie on IMDb Power Rangers Media TV series Saban Entertainment Mighty Morphin Power Rangers ( 1993 -- 1995 ; 2010 )
Season 1 2010 re-version Alien Rangers ( 1996 ) Zeo ( 1996 ) Episodes Turbo ( 1997 ) Episodes in Space ( 1998 ) Episodes Lost Galaxy ( 1999 ) Episodes
Lightspeed Rescue ( 2000 ) Episodes Time Force ( 2001 ) Episodes Disney Wild Force ( 2002 ) Episodes Ninja Storm ( 2003 ) Episodes Dino Thunder ( 2004 )
Episodes S.P.D. (
</item 5>

<item 6>
Mighty Morphin Power Rangers: The Movie
He felt that the characters , with the exception of Ivan Ooze , lacked personalities , and that the scenes of monsters rampaging through the city hearkened back
to the worst Japanese monster films . Mick LaSalle of the San Francisco Chronicle found the fights `` only adequately choreographed , '' called the battle in
the climax `` a complete disaster '' and stating that it made no sense in timing , that protagonists were not very intelligent , and the actors playing them
unremarkable . Home media ( edit ) This section does not cite any sources . Please help improve this section by adding citations to reliable sources . Unsourced
material may be challenged and removed . ( May 2015 ) ( Learn how and when to remove this template message ) The film was released on VHS and LaserDisc in late
1995 and then as a double feature with 1997 's Turbo : A Power Rangers Movie on a double - sided DVD in 2001 by 20th Century Fox Home Entertainment . Bonus
features included a theatrical trailer and a `` Making Of '' featurette . The film was then released separately on a single - sided DVD in 2003 .
</item 6>

<item 7>
Mighty Morphin Power Rangers: The Movie
Mighty Morphin Power Rangers : the Movie - wikipedia Mighty Morphin Power Rangers : the Movie Jump to : navigation , search `` Power Rangers : The Movie ''
redirects here . For the 2015 fan film , see Power / Rangers . For the 2017 film , see Power Rangers ( film ) .
</item 7>

<item 8>
Mighty Morphin Power Rangers: The Movie
Much like the television season that followed the release , it used concepts from the Japanese Super Sentai series Kyoryu Sentai Zyuranger , Ninja Sentai
Kakuranger . It is the first Power Rangers production from Saban Entertainment not to feature any archived footage from Super Sentai . Filming took place in and
around Sydney and Queensland , Australia . It was released by 20th Century Fox on June 30 , 1995 . The film grossed $66.43 million worldwide , but received
middling and polarizing reviews from critics . Contents ( hide ) 1 Plot 2 Cast 2.1 Voices 3 Music 4 Adaptations in other media 5 Release 5.1 Critical reception
6 Home media 7 References 8 External links Plot ( edit ) The Power Rangers participate with Bulk and Skull in a charity skydive for the Angel Grove observatory
, in anticipation of Ryan 's Comet which is scheduled to pass by in two days . Bulk and Skull miss the target landing zone and accidentally land on a
construction site where a giant egg has been unearthed .
</item 8>

<item 9>
Mighty Morphin Power Rangers: The Movie
When Fred Kelman , a friend of the Rangers ' , discovers his father missing , he finds him working at the construction site and discovers Ivan 's plans . On
Phaedos , the Rangers are almost killed by the Tengu , but are rescued by Dulcea , Phaedos ' Master Warrior . She initially tells them to leave for their own
safety , but after hearing of Zordon 's plight , she agrees to help them and takes them to an ancient ruined temple where the Rangers will have to overcome
obstacles to acquire the power of the Ninjetti . Dulcea awakens each Rangers ' animal spirit : Aisha is the bear , Rocky is the ape , Billy is the wolf ,
Kimberly is the crane , Adam is the frog and Tommy is the falcon . The Rangers make their way to the Monolith housing the Great Power , defeating its guardians
, and retrieve the Great Power . On Earth , Ivan 's Ecto - Morphicons are completely unearthed , and he unleashes them on Angel Grove , ordering the parents to
commit suicide at the construction site . Fred , Bulk , Skull and the other kids head to the construction site to save their parents .
</item 9>

<item 10>
Mighty Morphin Power Rangers: The Movie
( 2005 ) Episodes Mystic Force ( 2006 ) Episodes Operation Overdrive ( 2007 ) Episodes Jungle Fury ( 2008 ) Episodes RPM ( 2009 ) Episodes Saban Brands Samurai
& Super Samurai ( 2011 -- 2012 ) Season 1 Megaforce & Super Megaforce ( 2013 -- 2014 ) Season 1 Dino Charge & Dino Super Charge ( 2015 -- 2016 ) Season 1 Ninja
Steel & Super Ninja Steel ( 2017 -- 2018 ) Season 1 Beast Morphers ( 2019 ) Films Mighty Morphin Power Rangers : The Movie ( 1995 ) Turbo : A Power Rangers
Movie ( 1997 ) Saban 's Power Rangers ( 2017 ) Web series Power Rangers Hyperforce ( 2017 -- 2018 ) Video games Mighty Morphin Power Rangers ( 1994 ) The Movie
( 1995 ) The Fighting Edition ( 1995 ) Zeo : Battle Racers ( 1996 ) Lightspeed Rescue ( 2000 ) Time Force ( 2001 ) Wild Force ( 2002 ) Ninja Storm ( 2003 ) Dino
Thunder ( 2004
</item 10>
=== END ITEM id=634adc5108050f3a ===

=== ITEM id=dd8a00056fe3ab2e ===
Gold sentences and facts:
Sentence 0: In every High Court , there is a Chief Justice and many other judges whose number is defined by the President of India .
  Fact 0.0: Every High Court has a Chief Justice.
  Fact 0.1: Every High Court has many other judges.
  Fact 0.2: The number of other judges in a High Court is defined by the President of India.

Retrieved items:
<item 1>
List of Justices of the Supreme Court of Sri Lanka
450 -- Rajapaksa Sri Lanka Law College President of the Court of Appeal of Sri Lanka Sarath de Abrew 1953 -- 2016 7 May 2014 -- January 2016 -- -- Rajapaksa Sri
Lanka Law College Judge of the Court of Appeal Priyantha Jayawardena † 7 May 2014 -- present -- 7003145000000000000 ♠ 1,450 -- Rajapaksa Sri Lanka Law College
University of Aberdeen Upali Abeyratne † 17 December 2014 -- present -- 7003122600000000000 ♠ 1,226 -- Rajapaksa Judge of the Court of Appeal Anil Goonaratne †
30 January 2015 -- present -- 7003118200000000000 ♠ 1,
</item 1>

<item 2>
President of the United States
Administrative powers Suffice it to say that the President is made the sole repository of the executive powers of the United States , and the powers entrusted
to him as well as the duties imposed upon him are awesome indeed . William Rehnquist , Nixon v. General Services Administration , 433 U.S. 425 ( 1977 ) (
dissenting opinion ) The president is the head of the executive branch of the federal government and is constitutionally obligated to `` take care that the laws
be faithfully executed '' . The executive branch has over four million employees , including members of the military . Presidents make numerous executive branch
appointments : an incoming president may make up to 6,000 before taking office and 8,000 more while serving . Ambassadors , members of the Cabinet , and other
federal officers , are all appointed by a president with the `` advice and consent '' of a majority of the Senate . When the Senate is in recess for at least
ten days , the president may make recess appointments . Recess appointments are temporary and expire at the end of the next session of the Senate . The power of
a president to fire executive officials has long been a contentious political issue .
</item 2>

<item 3>
Indian High Courts Act 1861
It exercises the power of superintendence and control over all courts and tribunals throughout the territory except the military tribunals . Judicial tenure and
seniority ( edit ) Under § 5 of the Act , judges served at the pleasure of Her Majesty . The chief justice had precedence over judges , whereas judges had
seniority based on appointment . Under § 14 , the chief justice had the power to form divisions of two or more judges to hear cases . Jurisdiction ( edit )
Under article 9 of the Act , each High Court had `` all such powers and authority for and in relation to the administration of justice '' including original and
appellate jurisdiction over civil , criminal , admiralty , vice-admiralty , testamentary , intestate , and matrimonial matters . References ( edit ) Jump up ^
`` Indian High Courts Act 1861 '' . GKToday. 2011 - 10 - 17 . Retrieved 2017 - 03 - 20 . Jump up ^ Criminal Justice India Series : Punjab , 2002 . Allied
Publishers . 2002 . p. 233 .
</item 3>

<item 4>
Indian High Courts Act 1861
Contents ( hide ) 1 Abolishing Existing Courts 2 Qualifications of High Court Judges 3 Judicial Tenure and Seniority 4 Jurisdiction 5 References Abolishing
existing Courts ( edit ) The Act abolished the Supreme Courts at Calcutta , Madras , and Bombay ; the Sadar Diwani Adalat and the Sadar Nizamat Adalat at
Calcutta ; Sadar Adalat and Faujdari Adalat at Madras ; Sadar Diwani Adalat and Faujdari Adalat at Bombay ( § 8 ) . Qualifications of High Court judges ( edit )
Each High Court could consist of a chief justice and up to 15 judges . Under § 3 of the Act , judges could be selected from barristers ( with 5 years of
experience ) , civil servants ( with 10 years of experience including 3 years as a zillah judge ) , judges of small cause courts or sudder ameen ( with 5 years
of experience ) , or pleaders of lander courts or High Courts ( with 05 years of experience ) .
</item 4>

<item 5>
List of Justices of the Supreme Court of Sri Lanka
Imam Central 9 January 2009 -- 20 February 2013 -- 1,503 -- Rajapaksa Judge of the Court of Appeal R.K.S. Suresh Chandra 17 June 2010 -- -- -- Rajapaksa Sathya
Hettige 10 July 2011 -- 18 April 2014 -- 1,013 -- Rajapaksa University of Colombo President of the Court of Appeal of Sri Lanka Priyasath Dep † 10 July 2011 --
present 02 March 2017 - present -- 7003248200000000000 ♠ 2,482 -- Rajapaksa University of Colombo International Institute of Social Studies Solicitor General of
Sri Lanka Eva Wansundera † North Western 7 July 2012 -- present -- 7003211900000000000 ♠ 2,
</item 5>

<item 6>
Indian High Courts Act 1861
Qualifications for the Judges ( a ) He should be a citizen of India . ( b ) He should have been ( I ) a judge for 10 years of Subordinate court under the
Judicial Service of the State or ( ii ) an Advocate for 10 years in a High Courts in India ( Article 217 ) . Tenure : Originally the age of the retirement of
the judges of the High Courts was fixed at 60 but it was raised to 62 in 1963 according to the 15th amendment of the Constitution . Removal of the Judges : A
judge may leave his office by resigning . He will send his letter of resignation to the President . His office would be considered to have been vacated if he is
appointed as a judge of the Supreme Court or is transferred to some other High Court . A judge of a High Court may also be removed like a judge of the Supreme
Court . A judge of High Court may be removed by the President if the Parliament passes a motion against him by an absolute majority and 2 / 3rd majority of the
members present and voting , both the Houses sitting separately .
</item 6>

<item 7>
List of Justices of the Supreme Court of Sri Lanka
Justice of the Straits Settlements 35 George Henry Withers -- -- -- 36 Dodwell Browne -- -- -- 37 Fredrick Charles Moncrieff 4 April 1900 -- -- -- 38 Henry
Lorentz Wendt 15 October 1901 -- 28 November 1910 -- -- 39 John Page Middleton 28 January 1902 -- 1912 -- -- Puisine Judge of the Supreme Court of Cyprus 40
Charles Layard Western 26 April 1902 -- 18 June 1906 26 April 1902 -- 18 June 1906 1,514 1,514 41 Joseph Richard Grenier 28 November 1910 -- 1912 -- -- 42
Alexander Wood Renton 3 September 1905 -- 1918 22 August 1914 - 1918 -- 43 Alfred George Lascelles England 1 May 1911 -- 1914 1 May 1911 -- 1914 -- 44 Joseph
Turner Hutchinson England 23 October 1906 -- 1 May 1911 23 October 1906 -- 1 May 1911 1,
</item 7>

<item 8>
Indian High Courts Act 1861
Appellate Jurisdiction : The appellate jurisdiction of the High Courts extends so : ( a ) The High Court can hear appeals in civil cases if the amount involved
in the case is at least Rs. 5000 . ( b ) The High Court in criminal cases hears the appeal in which the accused has been sentenced to four years imprisonment by
the Sessions Judge . v ( c ) The death sentence awarded by Sessions Judge is subject to approval by the High Court . ( d ) The High Court hear the cases
involving interpretation of the Constitution or Law . ( e ) The High Court hears the cases on income tax , sales tax etc . Power of Judicial Review : The States
High Courts like the Supreme Court has the power of Judicial Review . A High Court has the power to strike down any law of the State or any order of the
executive if it violates any provision of the constitution or curtails or takes any of the Fundamental Rights of the people . Administrative and Supervisory
Power : The State High Court performs many administrative functions within its Territorial Jurisdiction .
</item 8>

<item 9>
Indian High Courts Act 1861
Salary : The pay of the Chief Justice of a High Court is rupees 90,000 / - per month and that of the other judges is rupees 85,000 / - per month . Powers and
Functions Original Jurisdiction : The original jurisdiction of the High Court is restricted . ( a ) Every High Court under Article 226 is empowered to issue
writs , orders , directions including writs in the nature of Habeas Corpus , Mandamus , Prohibition , Quo - warranto and Certiorari or any of them to any person
or authority with in its territory for the enforcement of the Fundamental Rights and for any other purpose . ( b ) The original jurisdiction of High Court
extends to matters of admiralty , matrimonial , contempt of court and cases ordered to be transferred to High Court by lower court . ( c ) The High Courts of
Mumbai , Kolkata and Chennai have original jurisdiction on hearing straightway cases involving the Christians and Parsies . ( d ) The High Courts of Mumbai ,
Kolkata and Chennai exercise original civil jurisdiction when the amount involved is more than two thousand rupees .
</item 9>

<item 10>
Indian High Courts Act 1861
The High Court of a State is the highest court of the State and all other courts of the State work under it . Normally there is one High Court in every State
but there can be only one High Court for two or more States as well , according to the constitution . There is one High Court at Chandigarh for Punjab , Haryana
and Union Territory of Chandigarh . Similarly there is one High Court at Guwahati which serves Assam , Arunachal Pradesh , Mizoram and Nagaland . Composition :
In every High Court , there is a Chief Justice and many other judges whose number is defined by the President of India . Appointment of the Judges : The Chief
Justice of a High Court is appointed by the President with the consultation of the Chief Justice of the Supreme Court and the Governor of the State . The other
judges are appointed by the will of President , Governor and the Chief Justice of High Court . Qualifications for the Judges ( a ) He should be a citizen of
India .
</item 10>
=== END ITEM id=dd8a00056fe3ab2e ===

=== ITEM id=f2398f05f926c6ad ===
Gold sentences and facts:
Sentence 0: Published Serialized in The National Era for ten months , beginning with the June 5 , 1851 issue . Published as a book in two volumes on March 20 , 1852 , John P. Jewett and Company .
  Fact 0.0: Uncle Tom's Cabin was serialized in The National Era for ten months.
  Fact 0.1: The serialization of Uncle Tom's Cabin in The National Era began with the June 5, 1851 issue.
  Fact 0.2: Uncle Tom's Cabin was published as a book in two volumes on March 20, 1852.
  Fact 0.3: The book edition of Uncle Tom's Cabin was published by John P. Jewett and Company.
Sentence 1: Uncle Tom 's Cabin first appeared as a 40 - week serial in The National Era , an abolitionist periodical , starting with the June 5 , 1851 , issue . It was originally intended as a shorter narrative that would run for only a few weeks . Stowe expanded the story significantly , however , and it was instantly popular , such that several protests were sent to the Era office when she missed an issue . Because of the story 's popularity , the publisher John P. Jewett contacted Stowe about turning the serial into a book . While Stowe questioned if anyone would read Uncle Tom 's Cabin in book form , she eventually consented to the request .
  Fact 1.0: Uncle Tom's Cabin first appeared as a 40-week serial in The National Era.
  Fact 1.1: The National Era was an abolitionist periodical.
  Fact 1.2: The serial started with the June 5, 1851 issue.
  Fact 1.3: Uncle Tom's Cabin was originally intended as a shorter narrative that would run for only a few weeks.
  Fact 1.4: Harriet Beecher Stowe expanded the story significantly.
  Fact 1.5: The expanded story was instantly popular.
  Fact 1.6: Several protests were sent to the National Era office when Stowe missed an issue.
  Fact 1.7: The publisher John P. Jewett contacted Stowe about turning the serial into a book because of the story's popularity.
  Fact 1.8: Stowe questioned whether anyone would read Uncle Tom's Cabin in book form.
  Fact 1.9: Stowe eventually consented to the request to publish the story as a book.
Sentence 2: Uncle Tom 's Cabin Uncle Tom 's Cabin , Boston edition Author Harriet Beecher Stowe Original title Uncle Tom 's Cabin ; or , Life Among the Lowly Illustrator Hammatt Billings ( 1st edition ) Country United States Language English Genre Novel Published Serialized in The National Era for ten months , beginning with the June 5 , 1851 issue . Published as a book in two volumes on March 20 , 1852 , John P. Jewett and Company . Followed by A Key to Uncle Tom 's Cabin ( 1853 )
  Fact 2.0: The infobox author of Uncle Tom's Cabin is Harriet Beecher Stowe.
  Fact 2.1: The original title of Uncle Tom's Cabin is Uncle Tom's Cabin; or, Life Among the Lowly.
  Fact 2.2: The illustrator of the 1st edition of Uncle Tom's Cabin is Hammatt Billings.
  Fact 2.3: Uncle Tom's Cabin is from the United States.
  Fact 2.4: Uncle Tom's Cabin is written in English.
  Fact 2.5: Uncle Tom's Cabin is a novel.
  Fact 2.6: Uncle Tom's Cabin was serialized in The National Era for ten months, beginning with the June 5, 1851 issue.
  Fact 2.7: Uncle Tom's Cabin was published as a book in two volumes on March 20, 1852, by John P. Jewett and Company.
  Fact 2.8: Uncle Tom's Cabin is followed by A Key to Uncle Tom's Cabin (1853).
  Fact 2.9: The infobox includes an image of the Boston edition of Uncle Tom's Cabin.

Retrieved items:
<item 1>
Uncle Tom's Cabin
The book was translated into all major languages , and in the United States it became the second best - selling book after the Bible . A number of the early
editions carried an introduction by Rev James Sherman , a Congregational minister in London noted for his abolitionist views . Uncle Tom 's Cabin sold equally
well in Britain , with the first London edition appearing in May 1852 and selling 200,000 copies . In a few years over 1.5 million copies of the book were in
circulation in Britain , although most of these were infringing copies ( a similar situation occurred in the United States ) . Plot Eliza escapes with her son ,
Tom sold `` down the River '' Full - page illustration by Hammatt Billings for Uncle Tom 's Cabin depicts Eliza telling Uncle Tom that he has been sold and she
is running away to save her child ( first edition : Boston : John P. Jewett and Company , 1852 ) . The book opens with a Kentucky farmer named Arthur Shelby
facing the loss of his farm because of debts .
</item 1>

<item 2>
Uncle Tom's Cabin
Uncle Tom 's Cabin - wikipedia Uncle Tom 's Cabin Jump to : navigation , search This article is about the mid-19th century novel . For other uses , see Uncle
Tom 's Cabin ( disambiguation ) . Uncle Tom 's Cabin Uncle Tom 's Cabin , Boston edition Author Harriet Beecher Stowe Original title Uncle Tom 's Cabin ; or ,
Life Among the Lowly Illustrator Hammatt Billings ( 1st edition ) Country United States Language English Genre Novel Published Serialized in The National Era
for ten months , beginning with the June 5 , 1851 issue . Published as a book in two volumes on March 20 , 1852 , John P. Jewett and Company .
</item 2>

<item 3>
Uncle Tom's Cabin
Uncle Tom 's Cabin : or Life among the lowly ; frontispiece by John Gilbert ; ornamental title - page by Phiz ; and 130 engravings on wood by Matthew Urlwin
Sears , 1853 ( a searchable facsimile at the University of Georgia Libraries ; DjVu & layered PDF format ) Pictures and stories from Uncle Tom 's Cabin ; `` The
purpose of the editor of this little work , has been to adapt it for the juvenile family circle . The verses have accordingly been written by the authoress for
the capacity of the youngest readers ... '' 1853 ( a searchable facsimile at the University of Georgia Libraries ; DjVu & layered PDF format ) Uncle Tom 's
Cabin at Project Gutenberg Uncle Tom 's Cabin , available at Internet Archive . Scanned , illustrated original editions .
</item 3>

<item 4>
Uncle Tom's Cabin
Convinced the book would be popular , Jewett made the unusual decision ( for the time ) to have six full - page illustrations by Hammatt Billings engraved for
the first printing . Published in book form on March 20 , 1852 , the novel sold 3,000 copies on that day alone , and soon sold out its complete print run . A
number of other editions were soon printed ( including a deluxe edition in 1853 , featuring 117 illustrations by Billings ) . In the first year of publication ,
300,000 copies of Uncle Tom 's Cabin were sold . At that point , however , `` demand came to an unexpected halt ... No more copies were produced for many years
, and if , as is claimed , Abraham Lincoln greeted Stowe in 1862 as ' the little woman who wrote the book that made this great war , ' the work had effectively
been out of print for many years . '' Jewett went out of business , and it was not until Ticknor and Fields put the work back in print in November 1862 that
demand began again to increase .
</item 4>

<item 5>
Uncle Tom's Cabin
ISSN 1551 - 5486 . ^ Jump up to : Henry Louis Gates , Kwame Anthony Appiah , Africana : Arts and Letters : An A-to - Z Reference of Writers , Musicians , and
Artists of the African American Experience , Running Press , 2005 , p. 544 . ^ Jump up to : Susan Logue , `` Historic Uncle Tom 's Cabin Saved '' , VOA News ,
January 12 , 2006 . Retrieved December 24 , 2011 . Jump up ^ Harriet Beecher Stowe , A Key to Uncle Tom 's Cabin 1853 , p. 42 , in which Stowe states : `` A
last instance parallel with that of Uncle Tom is to be found in the published memoirs of the venerable Josiah Henson ) ... '' This also is cited in Debra J.
Rosenthal , A Routledge Literary Sourcebook on Harriet Beecher Stowe 's Uncle Tom 's Cabin , Routledge , 2003 , pp. 25 -- 26 . Jump up ^ Annys Shin , `` After
buying historic home , Md.
</item 5>

<item 6>
Uncle Tom's Cabin
4 ( January 1960 ) , pp. 421 -- 433 . Jump up ^ The Classic Text : Harriett Beecher Stowe . University of Wisconsin - Milwaukee Library . Special collection
page on traditions and interpretations of Uncle Tom 's Cabin . Retrieved May 15 , 2007 . Archived May 16 , 2008 , at the Wayback Machine . ^ Jump up to : Uncle
Tom 's Cabin , introduction by Amanda Claybaugh , Barnes and Noble Classics , New York , 2003 , p. xvii . Jump up ^ A.M. Arnett , `` Review of James Baird
Weaver by Fred Emory Haynes '' , Political Science Quarterly , Vol. 35 , No. 1 ( March 1920 ) , pp. 154 -- 157 ; and profile of James Baird Weaver Archived
January 2 , 2007 , at the Wayback Machine. , accessed February 17 , 2007 . Jump up ^ Nassau Senior , quoted in Ephraim Douglass Adams , Great Britain and the
American Civil War ( 1958 ) , p. 33 .
</item 6>

<item 7>
Uncle Tom's Cabin
42 . ^ Jump up to : Thomas F. Gossett , `` Review of The Building of Uncle Tom 's Cabin by E. Bruce Kirkham '' , American Literature , Vol. 50 , No. 1 ( March
1978 ) , pp. 123 -- 124 . Jump up ^ Charles Nichols , `` The Origins of Uncle Tom 's Cabin '' , The Phylon Quarterly , Vol. 19 , No. 3 ( 3rd Qtr , 1958 ) , p.
328 . ^ Jump up to : Jane Tompkins , `` Sentimental Power : Uncle Tom 's Cabin and the Politics of Literary History '' Archived December 16 , 2007 , at the
Wayback Machine. , from In Sensational Designs : The Cultural Work of American Fiction , 1790 -- 1860 . New York : Oxford University Press , 1985 , pp. 122 --
146 . Jump up ^ Posner , R. , Public Intellectuals : A Study of Decline , Harvard University Press ( 2002 ) , p. 239 . ISBN 0 - 674 - 00633 - X .
</item 7>

<item 8>
Uncle Tom's Cabin
Simpson , Princeton University Press , 2001 . External links Wikimedia Commons has media related to Uncle Tom 's Cabin . Wikisource has original text related to
this article : Uncle Tom 's Cabin Uncle Tom 's Cabin on In Our Time at the BBC . University of Virginia Web site `` Uncle Tom 's Cabin and American Culture : A
Multi-Media Archive '' -- edited by Stephen Railton , covers 1830 to 1930 , offering links to primary and bibliographic sources on the cultural background ,
various editions , and public reception of Harriet Beecher Stowe 's influential novel . The site also provides the full text of the book , audio and video clips
, and examples of related merchandising .
</item 8>

<item 9>
Uncle Tom's Cabin
122 . Jump up ^ Gibson , Ian , The English Vice : Beating , Sex and Shame in Victorian England and After ( 1978 ) The passage which seems most likely to have
interested them is the start of chapter 29 . Jump up ^ `` La Cabaña del Tío Tomás '' ( PDF ) . Gaceta del Gobierno del Salvador en la América Central ( in
Spanish ) . Tomo 4 Número 17 . July 22 , 1853 . p. 4 . Archived from the original ( PDF ) on April 21 , 2018 . Retrieved April 21 , 2018 . Jump up ^ Tompkins ,
Jane . Sensational Designs : The Cultural Work of American Fiction , 1790 -- 1860 . New York : Oxford University Press , 1986 . See chapter five , ``
Sentimental Power : Uncle Tom 's Cabin and the Politics of Literary History . '' Jump up ^ Cindy Weinstein , The Cambridge Companion to Harriet Beecher Stowe ,
Cambridge University Press , 2004 , p. 13 .
</item 9>

<item 10>
Uncle Tom's Cabin
See also Twelve Years a Slave ( 1853 ) , bestselling narrative of free negro Solomon Northup published soon after Uncle Tom 's Cabin ( 1852 ) , which documents
facts supporting Stowe 's fictional narrative in detail , as the area where Northup was enslaved was close to the fictional setting of the plantation where much
of Stowe 's narrative takes place . History of slavery in the United States Origins of the American Civil War Ramona , an 1884 novel that attempted to do for
Native Americans in California what Uncle Tom 's Cabin had done for African Americans Riley - Bolten House , North Bethesda , Montgomery County , Maryland ,
plantation house of Isaac Riley listed on the National Register of Historic Places Timeline of the American Civil Rights Movement Collections Major collections
of Uncle Tom 's Cabin books , ephemera , and artifacts reside at the Albert and Shirley Small Special Collections Library at the University of Virginia and the
Harry Ransom Center at the University of Texas at Austin . Notes Jump up ^ Beecher Stowe , Harriet ( 1852 ) . Uncle Tom 's Cabin ; or , Life Among the Lowly .
I. Boston : John P. Jewitt .
</item 10>
=== END ITEM id=f2398f05f926c6ad ===

=== ITEM id=217dfdd1583f3bfb ===
Gold sentences and facts:
Sentence 0: `` A Whiter Shade of Pale '' is the debut single by the British rock band Procol Harum , released 12 May 1967 . The single reached number one in the UK Singles Chart on 8 June 1967 and stayed there for six weeks . Without much promotion , it reached number 5 on the Bilboard pop chart in the United States . One of the anthems of the 1967 Summer of Love , it is one of fewer than 30 singles to have sold over 10 million copies worldwide .
  Fact 0.0: "A Whiter Shade of Pale" is the debut single by the British rock band Procol Harum.
  Fact 0.1: "A Whiter Shade of Pale" was released on 12 May 1967.
  Fact 0.2: "A Whiter Shade of Pale" reached number one in the UK Singles Chart on 8 June 1967.
  Fact 0.3: "A Whiter Shade of Pale" stayed at number one in the UK Singles Chart for six weeks.
  Fact 0.4: Without much promotion, "A Whiter Shade of Pale" reached number 5 on the Billboard pop chart in the United States.
  Fact 0.5: "A Whiter Shade of Pale" is one of the anthems of the 1967 Summer of Love.
  Fact 0.6: "A Whiter Shade of Pale" is one of fewer than 30 singles to have sold over 10 million copies worldwide.

Retrieved items:
<item 1>
A Whiter Shade of Pale
You 're a Rich Man '' by The Beatles Procol Harum Gary Brooker Geoff Dunn Matt Pegg Josh Phillips Geoff Whitehorn Dave Ball Graham Broad Dave Bronze Pete Brown
Mark Brzezicki Alan Cartwright Chris Copping Matthew Fisher Mick Grabham Bobby Harrison David Knights Dee Murray Keith Reid Tim Renwick Ray Royer Don Snow Peter
Solley Henry Spinetti Jerry Stevenson Robin Trower Ian Wallace B.J.
</item 1>

<item 2>
A Whiter Shade of Pale
Its unusually allusive and referential lyrics are much more complex than most lyrics of the time ( for example , the chorus focuses on Chaucer 's `` The Miller
's Tale '' ) . Thus , this piece can be considered an early example of progressive rock . The phrase a whiter shade of pale has since gained widespread use in
the English language , noticed by several dictionaries . As such , the phrase is today often used in contexts independent of any consideration of the song . It
has also been heavily paraphrased , in forms like `` an Xer shade of Y '' , to the extent that it has been recognised as a snowclone -- a type of cliché and
phrasal template . Composition ( edit ) The song 's distinctive , Bach - inspired organ part was played on a Hammond M102 . The song is in moderate time in C
major and is characterised by the bassline moving stepwise downwards in a repeated pattern throughout . In classical music this is known as a ground bass . The
harmonic structure is identical for the organ melody , the verse and the chorus , except that the chorus finishes with a cadence .
</item 2>

<item 3>
A Whiter Shade of Pale
Contents ( hide ) 1 Recording and personnel 2 Lyrics 3 Composition 4 Release and reception 5 Promotional films 6 Authorship lawsuit 7 Chart performance 7.1
Weekly singles charts 7.2 Year - end charts 8 Annie Lennox version 8.1 Charts 9 References 10 Sources 11 External links Recording and personnel ( edit ) The
song was performed and recorded at Olympic Sound Studios in London , England , with Gary Brooker providing the vocals and piano , Matthew Fisher on a Hammond M
- 102 organ , David Knights on bass and Ray Royer on guitar . Drums were by session drummer Bill Eyden . A few days later , the song was re-recorded with the
band 's then newly recruited drummer Bobby Harrison , but that version was discarded , and one of the original mono recordings was chosen for release . Producer
for the record was Denny Cordell , and Keith Grant was the sound engineer . The song was included on the US release of the Procol Harum album , in September
1967 , but not on the later UK version . Lyrics ( edit ) Reid got the title and starting point for the song at a party .
</item 3>

<item 4>
A Whiter Shade of Pale
One of the anthems of the 1967 Summer of Love , it is one of fewer than 30 singles to have sold over 10 million copies worldwide . With its Bach - derived
instrumental melody , soulful vocals , and unusual lyrics -- by the song 's co-authors Gary Brooker , Keith Reid and Matthew Fisher -- `` A Whiter Shade of Pale
'' reached No. 1 in several countries when released in 1967 . In the years since , it has become an enduring classic . It was the most played song in the last
75 years in public places in the UK ( as of 2009 ) , and the United Kingdom performing rights group Phonographic Performance Limited in 2004 recognised it as
the most - played record by British broadcasting of the past 70 years . Also in 2004 , Rolling Stone placed `` A Whiter Shade of Pale '' 57th on its list of the
500 Greatest Songs of All Time . In 1977 , the song was named joint winner ( along with Queen 's `` Bohemian Rhapsody '' ) of `` The Best British Pop Single
1952 -- 1977 '' at the Brit Awards .
</item 4>

<item 5>
A Whiter Shade of Pale
The similarity is also referred to humorously in the 1982 play The Real Thing by Tom Stoppard and in the 1991 film The Commitments . Other writers have noted
similar `` family resemblances '' to other works by Bach : the Sinfonia which opens the cantata Ich steh mit einem Fuß im Grabe , BWV 156 , the organ chorale
Prelude `` O Mensch , bewein ' dein ' Sünde groß '' ( O Man , Lament Your Sin So Great ) , BWV 622 , from the Orgelbüchlein ( Little Organ Book ) and the
Cantata Sleepers Wake , BWV140 . The music also borrows ideas from `` When a Man Loves a Woman '' by Percy Sledge . Release and reception ( edit ) The single
was released on 12 May 1967 in the United Kingdom by Deram Records and entered Record Retailer 's chart ( later the UK Singles Chart ) on 25 May . In two weeks
it reached number 1 , where it stayed for six weeks .
</item 5>

<item 6>
A Whiter Shade of Pale
In his 1981 article on the musical and societal developments of 1967 , for The History of Rock , sociomusicologist Simon Frith described `` A Whiter Shade of
Pale '' as the year 's `` most distinctive single '' , through its combination of `` white soul vocal and a Bach organ exercise '' and enigmatic lyrics that ``
hinted at a vital secret open only to people in the right , drug - determined , state of mind '' . In the United States , the single reached number 5 on the
Billboard Hot 100 and sold over 1 million copies . It also peaked at number 22 on the soul charts there . In the Netherlands , the song entered the chart at
number 1 in June 1967 and again reached number 1 in July 1972 . A May 1972 re-release on Fly Records stayed in the UK charts for a total of 12 weeks and peaked
at number 13 . `` A Whiter Shade of Pale '' has continued to receive critical acclaim .
</item 6>

<item 7>
A Whiter Shade of Pale
Lyrics ( edit ) Reid got the title and starting point for the song at a party . He overheard someone at the party saying to a woman , `` You 've turned a whiter
shade of pale '' , and the phrase stuck in his mind . The original lyrics had four verses , of which only two are heard on the original recording . The third
verse has been heard in live performances by Procol Harum , and more seldom also the fourth . The author of Procol Harum : Beyond the Pale , Claes Johansen ,
suggests that the song `` deals in metaphorical form with a male / female relationship which after some negotiation ends in a sexual act '' . This is supported
by Tim de Lisle in Lives of the Great Songs , who remarks that the lyrics concern a drunken seduction , which is described through references to sex as a form
of travel , usually nautical , using mythical and literary journeys . Other observers have also commented that the lyrics concern a sexual relationship .
</item 7>

<item 8>
A Whiter Shade of Pale
`` A Whiter Shade of Pale '' has continued to receive critical acclaim . Along with Queen 's `` Bohemian Rhapsody '' , `` A Whiter Shade of Pale '' was jointly
recognised as `` The Best British Pop Single 1952 -- 1977 '' at the BRIT Awards , part of Elizabeth II 's Silver Jubilee . In 1998 the song was inducted into
the Grammy Hall of Fame . In 2004 , it appeared at number 57 on Rolling Stone magazine 's list of the `` 500 Greatest Songs of All Time '' . British TV station
Channel 4 placed the song at number 19 in its chart of `` The 100 Greatest No. 1 Singles '' . In 2018 , the song was inducted into the Rock and Roll Hall of
Fame in a new category for singles . Promotional films ( edit ) One of the clips for the song was filmed at Witley Court , part of which was gutted by fire in
1937 . The first promotional clip for `` A Whiter Shade of Pale '' was shot in the ruins of Witley Court in Worcestershire , England .
</item 8>

<item 9>
A Whiter Shade of Pale
A Whiter Shade of Pale - wikipedia A Whiter Shade of Pale Jump to : navigation , search `` A Whiter Shade of Pale '' A-side label of the UK vinyl single Single
by Procol Harum from the album Procol Harum B - side `` Lime Street Blues '' Released 12 May 1967 Format 7 - inch record Recorded Olympic Sound Studios , London
Genre Baroque pop Length 4 : 03 Label Deram Songwriter ( s ) Gary Brooker , Keith Reid , Matthew Fisher Producer ( s ) Denny Cordell Procol Harum singles
chronology `` A Whiter Shade of Pale '' ( 1967 ) `` Homburg '' ( 1967 ) `` A Whiter Shade of Pale '' ( 1967 ) `` Homburg '' ( 1967 ) `` A Whiter Shade of Pale
'' is the debut single by the British rock band Procol Harum , released 12 May 1967 . The single reached number one in the UK Singles Chart on 8 June 1967 and
stayed there for six weeks . Without much promotion , it reached number 5 on the Bilboard pop chart in the United States .
</item 9>

<item 10>
A Whiter Shade of Pale
live video from Eagle Rock Entertainment 's official channel on YouTube `` A Whiter Shade of Pale '' Scopitone film on YouTube Preceded by `` Silence is Golden
'' by The Tremeloes UK Singles Chart number one single 8 June 1967 -- 18 July 1967 ( six weeks ) Succeeded by `` All You Need Is Love '' by The Beatles Preceded
by `` Black Velvet Band '' by Johnny McEvoy Ireland 1967 number one single 22 June 1967 -- 19 July 1967 ( five weeks ) Succeeded by `` Black Velvet Band '' by
Johnny McEvoy Preceded by `` This is My Song '' by Petula Clark Australia 1967 number one single 8 July 1967 -- 28 July 1967 ( three weeks ) Succeeded by `` All
You Need is Love / Baby ,
</item 10>
=== END ITEM id=217dfdd1583f3bfb ===
