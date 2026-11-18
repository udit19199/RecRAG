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

This file has 11 items.

=== ITEM id=4d3004c3ac75741a ===
Gold sentences and facts:
Sentence 0: James David Sharman (born 12 March 1945), the son of boxing tent entrepreneur Jimmy Sharman, is an Australian director and writer for film and stage with more than 70 productions to his credit.
  Fact 0.0: James David Sharman was born on 12 March 1945.
  Fact 0.1: James David Sharman is the son of boxing tent entrepreneur Jimmy Sharman.
  Fact 0.2: James David Sharman is an Australian director.
  Fact 0.3: James David Sharman is an Australian writer for film and stage.
  Fact 0.4: James David Sharman has more than 70 productions to his credit.
Sentence 1: Ava Marie DuVernay ( ; born August 24, 1972) is an American director, screenwriter, film marketer, and film distributor.
  Fact 1.0: Ava Marie DuVernay was born on August 24, 1972.
  Fact 1.1: Ava Marie DuVernay is an American director.
  Fact 1.2: Ava Marie DuVernay is an American screenwriter.
  Fact 1.3: Ava Marie DuVernay is an American film marketer.
  Fact 1.4: Ava Marie DuVernay is an American film distributor.

Retrieved items:
<item 1>
Howard Barish
Howard Barish is an Oscar nominated, Emmy award winning producer and director known for his producing partnership with acclaimed director, writer Ava DuVernay.

 Their most recognized project to date, "13TH", is a 2016 American documentary from Netflix directed by DuVernay.

 Centered on race in the United States criminal justice system, the critically lauded film is titled after the Thirteenth Amendment to the United States
Constitution, which outlawed slavery (unless as punishment for a crime).

 It argues that slavery is being effectively perpetuated through mass incarceration.
</item 1>

<item 2>
Ava DuVernay
Ava Marie DuVernay ( ; born August 24, 1972) is an American director, screenwriter, film marketer, and film distributor.

 At the 2012 Sundance Film Festival, DuVernay won the Best Director Prize for her second feature film "Middle of Nowhere", becoming the first African-American
woman to win the award.

 For her work in "Selma" (2014), DuVernay was the first black female director to be nominated for a Golden Globe Award.

 With "Selma", she was also the first black female director to have her film nominated for the Academy Award for Best Picture.

 In 2017, she was nominated for the Academy Award for Best Documentary Feature for her film "13th" (2016).
</item 2>

<item 3>
A broad reference collection profiles international actors, opera singers, filmmakers, musicians, broadcasters, chefs, comedians, and other media figures,
alongside films, TV programs, awards, and radio formats. Highlights include Ava DuVernay’s Sundance win and Oscar-nominated *Selma* and *13th*; Hou
Hsiao-hsien’s major Cannes and Venice honors; acclaimed Israeli cinema; Mira Nair’s award-winning films; and competition-honored opera singers. It
</item 3>

<item 4>
♯JusticeForFlint
Initiated by "Creed" director Ryan Coogler and "Selma" director Ava DuVernay, the free public event was held at the Whiting Auditorium in Flint, Michigan.

 Hosted by comedian Hannibal Buress, it featured singers Janelle Monáe and Ledisi, as well as actor-activists Jesse Williams and Jussie Smollett, amongst
others.

 Surprising star guest was "King of Motown" Stevie Wonder, who teamed up with Monáe for "Hell You Talkin' Bout", a protest song she had written for the people
of Flint.
</item 4>

<item 5>
List of accolades received by Selma (film)
"Selma" is a 2014 American historical drama film directed by Ava DuVernay, and produced by Dede Gardner, Jeremy Kleiner, Christian Colson, and Oprah Winfrey.

 The screenplay was written by Paul Webb.

 The film follows the events leading up to and during the Selma to Montgomery marches, and the resulting establishment of the Voting Rights Act of 1965 which
prohibited racial discrimination in voting in the United States.

 The film stars David Oyelowo as civil rights activist Martin Luther King Jr., Tom Wilkinson as President Lyndon B. Johnson, and Tim Roth as Governor of Alabama
George Wallace.

 Carmen Ejogo, Dylan Baker, Wendell Pierce, Common, and Winfrey feature in supporting roles.
</item 5>

<item 6>
Selma (film)
Selma is a 2014 American historical drama film directed by Ava DuVernay and written by Paul Webb.

 It is based on the 1965 Selma to Montgomery voting rights marches led by James Bevel, Hosea Williams, Martin Luther King, Jr. and John Lewis.

 The film stars actors David Oyelowo as King, Tom Wilkinson as President Lyndon B. Johnson, Tim Roth as George Wallace, Carmen Ejogo as Coretta Scott King and
Common as Bevel.
</item 6>

<item 7>
Jim Sharman
James David Sharman (born 12 March 1945), the son of boxing tent entrepreneur Jimmy Sharman, is an Australian director and writer for film and stage with more
than 70 productions to his credit.

 He is renowned in Australia for his work as a theatre director from the 1960s to the present, and is best known internationally as the director of the 1973
theatrical hit "The Rocky Horror Show", its film adaptation "The Rocky Horror Picture Show" (1975) and the film's follow-up "Shock Treatment" (1981).
</item 7>

<item 8>
The entries profile filmmakers Pamela B. Green, Ava DuVernay, Howard Barish, Dana Brown, Matthew O’Neill, and the Maya Deren Award; actors Emayatzy Corinealdi
and Issa Rae; musicians Emel Mathlouthi, Siti binti Saad, and Angélique Kidjo; fitness figures Ingrid Romero and Anais Zanotti; and YouTubers Lilly Singh, Wayne
Goss, and Wengie. They
</item 8>

<item 9>
Ava DuVernay (born 1972) became the first African-American woman to win Sundance’s directing prize for Middle of Nowhere (2012). Selma (2014) brought historic
Golden Globe and Best Picture Oscar nominations; 13th (2016), examining race and mass incarceration, earned Oscar documentary nominations for DuVernay,
editor-producer Spencer Averick and producer Howard Barish. The entries also cover filmmakers Kathryn Bigelow and Steve McQueen, The
</item 9>

<item 10>
Spencer Averick
Spencer Averick is an American film editor and producer.

 Best known for his work an editor on critically acclaimed films "Middle of Nowhere" (2012), "Selma" (2014) and for producing 2016 acclaimed documentary "13th"
for which he received Academy Award for Best Documentary Feature nominations at 89th Academy Awards, that he shared with director Ava DuVernay and co-producer
Howard Barish.
</item 10>
=== END ITEM id=4d3004c3ac75741a ===

=== ITEM id=a2ba05c43b3aff31 ===
Gold sentences and facts:
Sentence 0:  He succeeded his brothers Eric IV Plovpenning and Abel of Denmark on the throne.
  Fact 0.0: The person in this sentence succeeded his brothers Eric IV Plovpenning and Abel of Denmark on the throne.
Sentence 1: Eric IV, also known as Eric Ploughpenny or Eric Plowpenny (Danish: "Erik Plovpenning" ), (  1216 – 10 August 1250) was king of Denmark from 1241 until his death in 1250.
  Fact 1.0: Eric IV is also known as Eric Ploughpenny.
  Fact 1.1: Eric IV is also known as Eric Plowpenny.
  Fact 1.2: Eric IV's Danish name is Erik Plovpenning.
  Fact 1.3: Eric IV's dates are 1216 to 10 August 1250.
  Fact 1.4: Eric IV was king of Denmark from 1241.
  Fact 1.5: Eric IV was king of Denmark until his death in 1250.

Retrieved items:
<item 1>
House of Oldenburg
The House of Oldenburg is a European dynasty of North German origin.

 It is one of Europe's most influential royal houses, with branches that rule or have ruled in Denmark, Iceland, Greece, Norway, Russia, Sweden, Schleswig,
Holstein, and Oldenburg.

 The current Queen of Denmark and King of Norway, the former King of Greece, the consorts of Greece and the United Kingdom, as well as the first twelve names in
the line of succession to the British throne, are all patrilineal members of the Glücksburg branch of this house.
</item 1>

<item 2>
Harald II of Denmark
Harald II of Denmark (died 1018) was King of Denmark from 1014 until his death in 1018.

 He was the youngest son of Sweyn Forkbeard and Gunhild of Wenden, and was regent while his father was fighting Ethelred the Unready in England.

 He inherited the Danish throne in 1014, and held it while his brother, the later king Cnut the Great conquered England.

 After his death in 1018, he was succeeded by Cnut the Great.

 Little detail is known about Harald II.
</item 2>

<item 3>
King Kang of Chu
King Kang of Chu (, died 545 BC) was from 559 to 545 BC the king of Chu, a major power during the Spring and Autumn period of ancient China.

 Born Xiong Zhao (), he succeeded his father King Gong of Chu.

 His reign was marked by constant wars with Jin, Chu's traditional enemy, and Wu, its new enemy.

 He ruled for 15 years and was succeeded by his son, Jia'ao, who would four years later be murdered by King Kang's younger brother King Ling of Chu.
</item 3>

<item 4>
John Christiansen
John “Chris” Christiansen (May 1, 1923 － September 12, 1998) was the chief military test pilot for Lockheed California Company for over 30 years.

 He might be most known for having performed Lockheed S-3 Viking's maiden flight on January 21, 1972.

 His assignments also included the initial test flights of Lockheed P-3 Orion.

 Christiansen was born in Oslo, Norway in 1923 and became an American citizen in 1939.

 He later served in the U.S. Navy during World War II and the Korean Conflict.

 He began experimental flying for Lockheed Martin in 1953, and worked there until his retirement in 1984.

 He was a fellow at Society of Experimental Test Pilots.
</item 4>

<item 5>
Eric IV of Denmark
Eric IV, also known as Eric Ploughpenny or Eric Plowpenny (Danish: "Erik Plovpenning" ), (  1216 – 10 August 1250) was king of Denmark from 1241 until his death
in 1250.

 He was the son of King Valdemar II by his wife, Berengaria of Portugal, and brother of King Abel and King Christopher I.
</item 5>

<item 6>
Christopher I of Denmark
Christopher I (Danish: "Christoffer I" ) (1219 – 29 May 1259) was King of Denmark between 1252 and 1259.

 He was the son of Valdemar II of Denmark by his wife, Infanta Berengária of Portugal.

 He succeeded his brothers Eric IV Plovpenning and Abel of Denmark on the throne.

 Christopher was elected King upon the death of his older brother Abel in the summer of 1252.

 He was crowned at Lund Cathedral on Christmas Day 1252.
</item 6>

<item 7>
Eric III of Denmark
Eric III Lamb (Danish: "Erik III Lam" ) (  1120 – 27 August 1146) was the King of Denmark from 1137 until 1146.

 He was the grandson of Eric I of Denmark and the nephew of Eric II of Denmark, whom he succeeded on the throne.

 He abdicated in 1146, as the first and only Danish monarch to do so.

 His succession led to a period of civil war between Sweyn III, Canute V, and Valdemar I.
</item 7>

<item 8>
Eric II of Denmark
Eric II the Memorable (Danish: "Erik II Emune" ) (  1090 – 18 September 1137) was king of Denmark between 1134 and 1137.

 Eric was an illegitimate son of Eric I of Denmark, who ruled Denmark from 1095 to 1103.

 Eric the Memorable rebelled against his uncle Niels of Denmark, and was declared king in 1134.

 He punished his adversaries severely, and rewarded his supporters handsomely.

 He was killed by a subject in 1137 and was promptly succeeded by his nephew Eric III of Denmark.
</item 8>

<item 9>
Prince Christian of Hanover (born 1985)
Prince Christian of Hanover (Christian Heinrich Clemens Paul Frank Peter Welf Wilhelm-Ernst Friedrich Franz Prince of Hanover, Duke of Brunswick and Lüneburg;
born 1 June 1985) is the younger son of Ernst August, Prince of Hanover, and his first wife Chantal Hochuli.

 He is the second in the line of succession to the former Hanoverian throne, after his elder brother Prince Ernst August.

 As a descendant of George III of the United Kingdom, Christian is also in the line of succession to the British throne.
</item 9>

<item 10>
Line of succession to the former Romanian throne
The succession order to the throne of the Romanian monarchy, abolished since 1947, was regulated by the monarchical constitution of 1923 and the 1884 Law of the
Romanian Royal House Rules enacted pursuant to the 1866 Constitution of Romania which had confirmed the enthronement of Prince Karl (Carol) of
Hohenzollern-Sigmaringen.

 The defunct 1923 Constitution stipulated Salic law, according to which the throne was hereditary in King Carol I's legitimate descent and, if his male issue
failed, in the descent of his brothers of the Sigmaringen princely branch of the House of Hohenzollern, according to agnatic primogeniture and to the perpetual
exclusion of females and their descendants.

 The last monarch to reign in Romania was King Michael I, who was born in 1921, abdicated his throne on 30 December 1947 under coercion, and went into exile in
Switzerland.
</item 10>
=== END ITEM id=a2ba05c43b3aff31 ===

=== ITEM id=a9659e02ea2fc41b ===
Gold sentences and facts:
Sentence 0: Martin Archer Flavin (November 2, 1883 – December 27, 1967) was an American playwright and novelist.
  Fact 0.0: Martin Archer Flavin was born on November 2, 1883.
  Fact 0.1: Martin Archer Flavin died on December 27, 1967.
  Fact 0.2: Martin Archer Flavin was an American playwright.
  Fact 0.3: Martin Archer Flavin was an American novelist.
Sentence 1: Peter Bradford Benchley (May 8, 1940 – February 11, 2006) was an American author and screenwriter.
  Fact 1.0: Peter Bradford Benchley was born on May 8, 1940.
  Fact 1.1: Peter Bradford Benchley died on February 11, 2006.
  Fact 1.2: Peter Bradford Benchley was an American author.
  Fact 1.3: Peter Bradford Benchley was an American screenwriter.

Retrieved items:
<item 1>
Martin Flavin (politician)
Martin Flavin (1841– 30 December 1916) was an Irish nationalist politician, butter merchant and prominent businessman from Cork.

 He was a Member of Parliament (MP) from 1891 to 1892 .
</item 1>

<item 2>
William Flynn Martin
William Flynn Martin (born October 4, 1950) is an American energy economist, educator and international diplomat.

 Martin served as Special Assistant to President Reagan for National Security Affairs, Executive Secretary of the National Security Council in the West Wing of
the White House and Deputy Secretary of the Department of Energy during the Ronald Reagan administration.

 He was President of the Council of the University for Peace, appointed to the Council by Secretary General of the United Nations Kofi Annan and served as the
Executive Director of the Republican Platform Committee during the re-election bid of George H.W. Bush.

 He has held senior appointments and advisory positions under several Presidents including: Ronald Reagan, George H.W. Bush and George W. Bush.
</item 2>

<item 3>
Mark Benton
Mark Benton (born 16 November 1965) is an English actor, known for his roles as Eddie in "Early Doors", Howard in "Northern Lights" and Martin Pond in
"Barbara".

 Benton has also starred in BBC One drama, "Waterloo Road" as maths teacher, Daniel Chalk from 2011 to 2013 and again in 2014.
</item 3>

<item 4>
Jerry Beck
Jerry Beck (born February 9, 1955 in New York City) is an American animation historian, author, blogger, and video producer.

 The author and/or editor of several books on classic American animation and classic characters, including "The 50 Greatest Cartoons" (1994), "The Animated
Movie Guide" (2005), "Not Just Cartoons: Nicktoons!"

 (2007), "The Flintstones: The Official Guide to the Cartoon Classic" (2011), "The Hanna-Barbera Treasury: Rare Art Mementos from Your Favorite Cartoon
Classics" (2007), "The SpongeBob SquarePants Experience: A Deep Dive into the World of Bikini Bottom" (2013), "Pink Panther: The Ultimate Guide" (2005), and
"Looney Tunes and Merrie Melodies: A Complete Illustrated Guide to the Warner Bros.

 Cartoons" (with Will Friedwald, 1989) alongside "The 100 Greatest Looney Tunes Cartoons" (2010).

 He is also an authority on the making of modern films, with his books detailing the art of "Mr.
</item 4>

<item 5>
Peter Benchley
Peter Bradford Benchley (May 8, 1940 – February 11, 2006) was an American author and screenwriter.

 He is known for the bestselling novel "Jaws" and co-wrote its subsequent film adaptation with Carl Gottlieb.

 Several more of his works were also adapted for cinema, including "The Deep", "The Island", "Beast", and "White Shark".
</item 5>

<item 6>
John Biddle (yachting cinematographer)
John Scott Biddle (1925–2008) was a foremost yachting cinematographer and lecturer, establishing a film-making career that spanned more than forty years.

 His films captured not only the technical aspects of sailing but also the human story in events as tranquil as a Nova Scotia cruise and as grand as the
America's Cup Races.
</item 6>

<item 7>
Michael Breed
Michael S. Breed (born May 14, 1962, in Greenwich, Connecticut) is a professional golf instructor and television host.

 In 2003, he was selected as a Top 100 Instructor in America by "Golf Magazine", in 2011, he was voted one of the Top 50 Instructors in America by Golf Digest
(now 13th on this list, and #1 in the state of New York.)

, in 2012, Breed was chosen as the PGA's National Teacher of the Year.

 Prior to his time as a television host, Breed was the Head Golf Professional at Sunningdale Country Club for 12 years from 2001–2012, and the Head Golf
Professional at Birchwood Country Club, as well as the Assistant Professional at Deepdale Golf Club and Augusta National Golf Club.

 He has served on different boards for the Metropolitan PGA for over 10 years and has represented the PGA of America at the National Golf Day in Washington D.C.
since 2012.

 He is involved in charities such as The First Tee, Folds of Honor, Hope for the Warriors, Wounded Warriors Foundation and Salute Military Golf Association.
</item 7>

<item 8>
Journey in the Dark
Journey in the Dark is a 1943 novel by Martin Flavin.

 It won both the 1943 Harper Prize and the 1944 Pulitzer Prize.
</item 8>

<item 9>
Dan Flannery
Arthur Daniel Flannery (born March 10, 1944) is an American actor from Evanston, Illinois.

 He has played roles in several movies, short films, and T.V. shows from a variety of genres including drama, action, and thriller.

 His filmography includes The Straight Story (1999), and Contagion (2011) and T.V. series such as Empire (2015 TV series) and Boss (2011-2012).
</item 9>

<item 10>
Martin Flavin
Martin Archer Flavin (November 2, 1883 – December 27, 1967) was an American playwright and novelist.
</item 10>
=== END ITEM id=a9659e02ea2fc41b ===

=== ITEM id=2e508b7b45653bbe ===
Gold sentences and facts:
Sentence 0: Mirpur University of Science & Technology (میرپور یونیورسٹی براۓ سائنس اور ٹیکنولوجی) (MUST) was formerly a constituent college of University of Azad Jammu and Kashmir as University College of Engineering & Technology Mirpur (UCET Mirpur).
  Fact 0.0: Mirpur University of Science & Technology (MUST) was formerly a constituent college of University of Azad Jammu and Kashmir.
  Fact 0.1: The Urdu name of Mirpur University of Science & Technology is میرپور یونیورسٹی براۓ سائنس اور ٹیکنولوجی.
  Fact 0.2: Mirpur University of Science & Technology was formerly named University College of Engineering & Technology Mirpur (UCET Mirpur).
Sentence 1: The University of Debrecen (Hungarian: "Debreceni Egyetem" ) is a university located in Debrecen, Hungary.
  Fact 1.0: The University of Debrecen is a university located in Debrecen, Hungary.
  Fact 1.1: The Hungarian name of the University of Debrecen is 'Debreceni Egyetem'.

Retrieved items:
<item 1>
University of Delaware
The University of Delaware (colloquially "UD") is the largest university in Delaware.

 The main campus is in Newark, with satellite campuses in Dover, Wilmington, Lewes, and Georgetown.

 It is considered a large institution with approximately 18,500 undergraduate and 4,500 graduate students. UD is a privately governed university which receives
public funding for being a land-grant, sea-grant, space-grant and urban-grant state-supported research institution.
</item 1>

<item 2>
Dalhousie University
Dalhousie University (commonly known as Dal) is a public research university in Nova Scotia, Canada, with three campuses in Halifax, a fourth in Bible Hill, and
medical teaching facilities in Saint John, New Brunswick.

 Dalhousie offers more than 4,000 courses and 180 degree programs in twelve undergraduate, graduate, and professional faculties.

 The university is a member of the U15, a group of research-intensive universities in Canada.
</item 2>

<item 3>
The text profiles diverse educational and related institutions worldwide. Highlights include Berat University (Albania), founded in 2009 and expanding to four
faculties; Rotman (Toronto), established in 1950; Auburn, Arizona, Debrecen, Copenhagen, Dalhousie and other research universities; specialist schools such as
Falmouth and Aalborghus; and private or state colleges across Ghana, Malaysia, Norway, Pakistan and elsewhere. It also covers Syracuse sororities, a
</item 3>

<item 4>
Mirpur University of Science and Technology
Mirpur University of Science & Technology (میرپور یونیورسٹی براۓ سائنس اور ٹیکنولوجی) (MUST) was formerly a constituent college of University of Azad Jammu and
Kashmir as University College of Engineering & Technology Mirpur (UCET Mirpur).

 It is a state university and the President of Azad Jammu & Kashmir is the Chancellor of the university.

 The Vice-Chancellor is the executive head and manages the university functions.
</item 4>

<item 5>
University of Debrecen
The University of Debrecen (Hungarian: "Debreceni Egyetem" ) is a university located in Debrecen, Hungary.

 It is the oldest continuously operating institution of higher education in Hungary (since 1538).

 The university has a well established programme in the English language for international students, particularly in the Medical field, which first established
education in English in 1986.

 There are nearly 4000 international students studying at the university.
</item 5>

<item 6>
Bangladesh University of Business and Technology
Bangladesh University of Business and Technology (Bengali: বাংলাদেশ ইউনিভার্সিটি অফ বিজনেস অ্যান্ড টেকনোলজি ) or BUBT is one of the eight private universities
to receive the green Signal from the government, located in Mirpur, Dhaka, Bangladesh.

 The university was established under the Private University Act 1992.

 BUBT is regulated by the Bangladesh University Grants Commission (UGC).
</item 6>

<item 7>
Albanian University in Berat
The University of Berat is a private institution, independent and secular Higher Education, which operates the state fee.

 Licensed by the Council of Ministers, no. 697 dated 18.06.2009, he opened the doors in the academic year 2009- 2010, offering programs in two faculties:
Faculty of Social Sciences Programs in Science Teaching Legal, Political Science and Psychology General Administration and the Faculty of Economics with
Learning programs in Banking and Finance-Business Administration.

 During the first academic year, about 1,000 students attended studies.

 A year later, this new institution offering undergraduate degrees in study programs: Preschool Teacher Education, Primary Education Teacher, English Language,
Nursing, Engineering Computer and Information Technology, thus finalizing the 2010-2011 academic year with the opening of two new faculties: Faculty of Medical
Sciences and the Faculty of Architecture and Inxhinierive.

 Academic Year 2010-2011 finds the University Berat with about 2600 students in the first cycle of study.

 Since the first year of its inception, the institution pointed at the center of academic life, the promotion of intellectual product and melting learning and
research.
</item 7>

<item 8>
Albanian University in Berat
Through presentations, research and exchange, the University directs and encourages the identification and development of projects that will once again
consolidate its primary purpose, scientific development, and research toward better quality.
</item 8>

<item 9>
The entries cover diverse schools, hospitals, and community institutions. They include Jesuit and other schools in the U.S., Sweden, India, Nepal, Belize, and
Japan-related programs in the U.S.; several Indian schools are fully residential, while Japanese supplementary schools serve expatriate children. U.S. district
profiles describe grade structures and administrators. Health organizations include historic and regional hospitals, Adventist HealthCare, and Common Hope’s
support for Guatemalan children. Other sites include St. John Cemetery
</item 9>

<item 10>
Education in Hyderabad
Hyderabad is an important seat of learning in southern India.

 The city hosts two central universities, three deemed universities, and six state universities.

 Osmania University is one of the oldest universities in India.

 Many institutes for education like University of Hyderabad, Jawaharlal Nehru Technological University, Hyderabad, International Institute of Information
Technology, Indian Institute of Technology, Hyderabad, Tata Institute of Fundamental Research,Hyderabad, Icfai University, Birla Institute of Technology and
Science, Pilani – Hyderabad and medical colleges like Nizam's Institute of Medical Sciences are located there.

 Also based in the city are the Institute of Public Enterprise and the National Academy of Legal Studies & Research (NALSAR).

 Hyderabad has various research institutes such as the Indian Institute of Chemical Technology, Centre for Cellular and Molecular Biology.

 It is also the home of Maulana Azad National Urdu University as well as Dr. B.R. Ambedkar Open University.

 This educational infrastructure attracts students from all over India and some international students (especially from Africa and the Middle East).

 Most prolific business school in India Nmims is also expanding its roots in Hyderabad.
</item 10>
=== END ITEM id=2e508b7b45653bbe ===

=== ITEM id=9ab22b8a257afd46 ===
Gold sentences and facts:
Sentence 0: The 2015 Africa Magic Viewers Choice Awards (AMVCA) was held on March 7, 2015 at the Eko Hotel and Suites, Victoria Island, Lagos, Nigeria.
  Fact 0.0: The 2015 Africa Magic Viewers Choice Awards (AMVCA) was held.
  Fact 0.1: The 2015 AMVCA was held on March 7, 2015.
  Fact 0.2: The 2015 AMVCA was held at the Eko Hotel and Suites.
  Fact 0.3: The Eko Hotel and Suites is in Victoria Island, Lagos, Nigeria.
Sentence 1: Lagos initially emerged as a port city which originated on a collection of islands, which are contained in the present day Local Government Areas (LGAs) of Lagos Island, Eti-Osa, Amuwo-Odofin and Apapa; the islands are separated by creeks, fringing the southwest mouth of Lagos Lagoon, while protected from the Atlantic Ocean by barrier islands and long sand spits such as Bar Beach, which stretch up to 100 km east and west of the mouth.
  Fact 1.0: Lagos initially emerged as a port city.
  Fact 1.1: Lagos originated on a collection of islands.
  Fact 1.2: The islands are contained in the present day Local Government Areas (LGAs) of Lagos Island, Eti-Osa, Amuwo-Odofin and Apapa.
  Fact 1.3: The islands are separated by creeks.
  Fact 1.4: The creeks fringe the southwest mouth of Lagos Lagoon.
  Fact 1.5: The islands are protected from the Atlantic Ocean by barrier islands.
  Fact 1.6: The islands are protected from the Atlantic Ocean by long sand spits such as Bar Beach.
  Fact 1.7: The barrier islands and long sand spits, such as Bar Beach, stretch up to 100 km east and west of the mouth.

Retrieved items:
<item 1>
Femi Jacobs
Femi Jacobs (born Oluwafemisola Jacobs; 8 May) is a Nigerian actor, speaker and singer.

 He came into prominence for playing Makinde Esho in the film "The Meeting", which also stars Rita Dominic and Jide Kosoko.

 For his role in "The Meeting", he received a nomination for Best Actor in a Leading Role at the 9th Africa Movie Academy Awards.

 He also won the award for Best Actor in a Comedy at the 2015 Africa Magic Viewers Choice Awards (AMVCA).
</item 1>

<item 2>
Lagos
Lagos initially emerged as a port city which originated on a collection of islands, which are contained in the present day Local Government Areas (LGAs) of
Lagos Island, Eti-Osa, Amuwo-Odofin and Apapa; the islands are separated by creeks, fringing the southwest mouth of Lagos Lagoon, while protected from the
Atlantic Ocean by barrier islands and long sand spits such as Bar Beach, which stretch up to 100 km east and west of the mouth.

 Due to rapid urbanization, the city expanded to the west of the lagoon to include areas in the present day Lagos Mainland, Ajeromi-Ifelodun and Surulere.

 This led to the classification of Lagos into two main areas - the Island, which was the initial city of Lagos, before it expanded into the area known as the
Mainland.

 This city area was governed directly by the Federal Government through the "Lagos City Council", until the creation of Lagos State in 1967, which led to the
splitting of Lagos city into the present day seven Local Government Areas (LGAs), and an addition of other towns (which now make up 13 LGAs) from the then
Western Region, to form the state.
</item 2>

<item 3>
OC Ukeje
Okechukwu Ukeje, known as OC Ukeje is Nigerian actor, model and musician.

 He came into prominence after winning the Amstel Malta Box Office (AMBO) reality show.

 He has received several awards including Africa Movie Academy Awards, Africa Magic Viewers Choice Awards, Nollywood Movies Awards, Best of Nollywood Awards,
Nigeria Entertainment Awards and Golden Icons Academy Movie Awards.

 He has featured in several award winning films including "Two Brides and a Baby", "Hoodrush", "Alan Poza", "Confusion Na Wa" and "Half of a Yellow Sun".
</item 3>

<item 4>
British Academy Television Awards 2009
The 2009 British Academy Television Awards were held on 26 April at the Royal Festival Hall in London.

 The event was broadcast live on BBC One and was hosted by Graham Norton.

 The nominations were announced on 24 March.

 Winners in bold.
</item 4>

<item 5>
2013 Africa Magic Viewers Choice Awards
The 2013 Africa Magic Viewers Choice Awards were held on 9 March 2014 and co-hosted by Big Brother Africa presenter IK Osakioduwa and StarGist’s host, Vimbai
Mutinhiri.

 Ivie Okujaiye and Olu Jacobs were honored with the TrailBlazer and Industry Merit Awards respectively.
</item 5>

<item 6>
Africa Magic Viewers' Choice Awards
Africa Magic Viewers' Choice Awards (AMVCA) is an annual accolade presented by Multichoice recognizing outstanding achievement in television and film, voted on
by the general public.

 The inaugural Africa Magic Viewers’ Choice Awards ceremony was held in Lagos, Lagos State in Nigeria on 9 March 2013, and was broadcast live in more than 50
countries.

 Entries into the award ceremony are films and TV series that have been aired on Dstv Channels in the previous year.
</item 6>

<item 7>
2015 Africa Magic Viewers Choice Awards
The 2015 Africa Magic Viewers Choice Awards (AMVCA) was held on March 7, 2015 at the Eko Hotel and Suites, Victoria Island, Lagos, Nigeria.

 IK and Vimbai were the hosts of the event.

 A new category, Best Indigenous Language (Igbo), was introduced in the 2015 awards.
</item 7>

<item 8>
Two Brides and a Baby
Two Brides and a Baby is a 2011 Nigerian romantic drama film directed by Teco Benson, starring Keira Hewatch, Kalu Ikeagwu, OC Ukeje, Chelsea Eze, Stella
Damasus-Aboderin and Okey Uzoeshi.

 It premiered on November 17, 2011.

 It received awards and nominations at Africa Movie Academy Awards, Best of Nollywood Awards and Africa Magic Viewers Choice Awards.
</item 8>

<item 9>
Africa Magic
Africa Magic is a collection of Pay TV entertainment channels that focus on African programming.

 "Africa Magic", which started off as single channel of the same name, is a brand owned by M-Net and now comprises eight channels.

 The first "Africa Magic" channel was launched in July 2003 as a movie channel and over the next decade, the brand expanded to include seven more channels
comprising movies, television shows and general entertainment.

 Africa Magic currently broadcasts in more than 50 African countries.

 The channels include "Africa Magic Family", "Africa Magic World", "Africa Magic Showcase", "Africa Magic Yoruba", "Africa Magic Igbo", "Africa Magic Hausa".

 "Africa Magic Epic" and "Africa Magic Urban".

 Africa Magic is also responsible for the annual Africa Magic Viewers' Choice Awards (AMVCAs), the biggest celebration of film and television talent in Africa.
</item 9>

<item 10>
Eric Aghimien
Eric Enomamien Aghimien is a Nigerian director, producer, screenwriter and editor.

 His debut feature film, "A Mile from Home" won awards at both the 2014 Africa Magic Viewers Choice Awards and the 10th Africa Movie Academy Awards.
</item 10>
=== END ITEM id=9ab22b8a257afd46 ===

=== ITEM id=110e754f15d8dd07 ===
Gold sentences and facts:
Sentence 0: McFarland, USA (also known as McFarland) is a 2015 American sports drama film directed by Niki Caro, produced by Mark Ciardi and Gordon Gray, written by Christopher Cleveland, Bettina Gilois and Grant Thompson with music composed by Antônio Pinto.
  Fact 0.0: McFarland, USA is also known as McFarland.
  Fact 0.1: McFarland, USA is a 2015 American sports drama film.
  Fact 0.2: McFarland, USA was directed by Niki Caro.
  Fact 0.3: McFarland, USA was produced by Mark Ciardi.
  Fact 0.4: McFarland, USA was produced by Gordon Gray.
  Fact 0.5: McFarland, USA was written by Christopher Cleveland.
  Fact 0.6: McFarland, USA was written by Bettina Gilois.
  Fact 0.7: McFarland, USA was written by Grant Thompson.
  Fact 0.8: McFarland, USA has music composed by Antônio Pinto.
Sentence 1: The Black Hole is a 1979 American science fiction film directed by Gary Nelson and produced by Walt Disney Productions.
  Fact 1.0: The Black Hole is a 1979 American science fiction film.
  Fact 1.1: The Black Hole was directed by Gary Nelson.
  Fact 1.2: The Black Hole was produced by Walt Disney Productions.

Retrieved items:
<item 1>
The Black Hole
The Black Hole is a 1979 American science fiction film directed by Gary Nelson and produced by Walt Disney Productions.

 The film stars Maximilian Schell, Robert Forster, Joseph Bottoms, Yvette Mimieux, Anthony Perkins, and Ernest Borgnine, while the voices of the main robot
characters are provided by Roddy McDowall and Slim Pickens (both unbilled).

 The music for the film was composed by John Barry.

 It was the first film from Walt Disney Productions to receive a PG rating.

 The film was released on December 18, 1979 in the United Kingdom and on December 21, 1979 in the United States.
</item 1>

<item 2>
Steve McQueen (director)
Steven Rodney "Steve" McQueen (born 9 October 1969) is an English film director, producer, screenwriter, and video artist.

 For his 2013 film, "12 Years a Slave", a historical drama adaptation of an 1853 slave narrative memoir, he won an Academy Award, BAFTA Award for Best Film, and
Golden Globe Award for Best Motion Picture – Drama, as a producer, and he also received the award for Best Director from the New York Film Critics Circle.

 McQueen is the first black filmmaker to win an Academy Award for Best Picture.

 McQueen is known for his collaborations with actor Michael Fassbender, who has starred in all three of McQueen's feature films as of 2014.

 McQueen's other feature films are "Hunger" (2008), a historical drama about the 1981 Irish hunger strike, and "Shame" (2011), a drama about an executive
struggling with sex addiction.
</item 2>

<item 3>
Full Metal Jacket
Full Metal Jacket is a 1987 British-American war film directed and produced by Stanley Kubrick.

 The screenplay by Kubrick, Michael Herr, and Gustav Hasford was based on Hasford's novel "The Short-Timers" (1979).

 Its storyline follows a platoon of U.S. Marines through their training, primarily focusing on two privates, Joker and Pyle, who struggle to get through camp
under their foul-mouthed drill instructor, Gunnery Sergeant Hartman, and the experiences of two of the platoon's Marines in the Tet Offensive during the Vietnam
War.

 The film's title refers to the full metal jacket bullet used by soldiers.

 The film was released in the United States on June 26, 1987.
</item 3>

<item 4>
McFarland, USA
McFarland, USA (also known as McFarland) is a 2015 American sports drama film directed by Niki Caro, produced by Mark Ciardi and Gordon Gray, written by
Christopher Cleveland, Bettina Gilois and Grant Thompson with music composed by Antônio Pinto.

 The film was co-produced by Walt Disney Pictures and Mayhem Pictures.

 Based on the true story of a 1987 cross country team from a mainly Latino high school in McFarland, California, the film stars Kevin Costner as Jim White, the
school's coach, who leads the team to win a state championship.

 The film also stars Maria Bello and Morgan Saylor.
</item 4>

<item 5>
Million Dollar Arm
Million Dollar Arm is a 2014 American biographical sports drama film directed by Craig Gillespie and produced by Walt Disney Pictures from a screenplay written
by Thomas McCarthy.

 The film is based on the true story of baseball pitchers Rinku Singh and Dinesh Patel who were discovered by sports agent J. B. Bernstein after winning a
reality show competition.
</item 5>

<item 6>
List of accolades received by The Hurt Locker
"The Hurt Locker" is a 2009 Iraq War film written by Mark Boal and directed by Kathryn Bigelow.

 The film premiered on September 4, 2008 at the 65th Venice International Film Festival, where it competed with "The Wrestler" for the Golden Lion award.

 It was released in Italy by Warner Bros.

 Pictures on October 10, 2008.

 Following a showing at the 33rd Toronto International Film Festival, Summit Entertainment picked the film up for distribution in the United States.

 "The Hurt Locker" was released in the United States on June 26, 2009, with a limited release at four theaters in Los Angeles and New York City.

 The film grossed over $145,000 on its opening weekend, averaging around $36,000 per theater.

 The following weekend, beginning July 3, the film grossed over $131,000 at nine theaters.

 It held the highest per-screen average of any movie playing theatrically in the United States for the first two weeks of its release, gradually moving into the
top 20 chart.

 Summit Entertainment then released "The Hurt Locker" to more screens.
</item 6>

<item 7>
Remember the Titans
Remember the Titans is a 2000 American sports film produced by Jerry Bruckheimer and directed by Boaz Yakin.

 The screenplay, written by Gregory Allen Howard, is based on the true story of African-American coach Herman Boone, portrayed by Denzel Washington, and his
attempt to integrate the T. C. Williams High School football team in Alexandria, Virginia, in 1971.

 Will Patton portrays Bill Yoast, Boone's assistant coach.

 Real-life athletes Gerry Bertier and Julius Campbell are portrayed by Ryan Hurst and Wood Harris, respectively.
</item 7>

<item 8>
List of American films of 2004
A list of American films released in 2004. "

Million Dollar Baby" won the 2004 Academy Award for Best Picture and "Crash" won the 2005 Academy Award for Best Picture. "

The Aviator" won the BAFTA Award for Best Film and the Golden Globe Award for Best Motion Picture – Drama. "

Sideways" won the Golden Globe Award for Best Motion Picture – Musical or Comedy and the Satellite Award for Best Film – Musical or Comedy. "

Hotel Rwanda" won the Satellite Award for Best Film – Drama.
</item 8>

<item 9>
Dangal (film)
Dangal (English: "Wrestling competition" ) is a 2016 Indian Hindi-language biographical sports drama film loosely based on the story of Mahavir Singh Phogat, an
amateur wrestler, who trains his daughters Geeta Phogat and Babita Kumari to be world-class wrestlers.

 It was directed by Nitesh Tiwari.

 Produced by Walt Disney Pictures, Aamir Khan Productions and UTV Motion Pictures, it stars Aamir Khan as the father.

 Both daughters go on to win medals at the 2010 Commonwealth Games.

 Babita won a silver medal at the Games, in the 51 kg class and gold at the 2014 Glasgow Games in the 55 kg class.

 In 2012, Geeta became the first Indian female wrestler to qualify for the Olympics.

 Mahavir's efforts inspired dozens of Indian women to take to wrestling.

 Fatima Sana Shaikh and Sanya Malhotra play the older selves of the sisters while Sakshi Tanwar plays their mother and Aparshakti Khurana, their cousin.
</item 9>

<item 10>
Kathryn Bigelow
Kathryn Ann Bigelow ( ; born November 27, 1951) is an American director, producer, and writer.

 Her films include the vampire Western horror film "Near Dark" (1987), the action crime film "Point Break" (1991), the science fiction action thriller "Strange
Days" (1995), the mystery thriller "The Weight of Water" (2000), the submarine thriller "" (2002), the war film "The Hurt Locker" (2008), the action thriller
war film "Zero Dark Thirty" (2012), the short film "Last Days" (2014), and the period crime drama "Detroit" (2017).

 "The Hurt Locker" won the 2009 Academy Award for Best Picture and the BAFTA Award for Best Film, and was nominated for the 2010 Golden Globe Award for Best
Drama.

 She has also acted as producer and writer for many of her films.
</item 10>
=== END ITEM id=110e754f15d8dd07 ===

=== ITEM id=8a4ac4853625f377 ===
Gold sentences and facts:
Sentence 0: Isaac Asimov ( ; born Isaak Ozimov; January 2, 1920 – April 6, 1992) was a Russian-American writer and professor of biochemistry at Boston University.
  Fact 0.0: Isaac Asimov was born Isaak Ozimov.
  Fact 0.1: Isaac Asimov was born on January 2, 1920.
  Fact 0.2: Isaac Asimov's dates run from January 2, 1920 to April 6, 1992.
  Fact 0.3: Isaac Asimov was a Russian-American writer.
  Fact 0.4: Isaac Asimov was a professor of biochemistry at Boston University.
Sentence 1: Michael Cunningham (born November 6, 1952) is a U.S. novelist and screenwriter.
  Fact 1.0: Michael Cunningham was born on November 6, 1952.
  Fact 1.1: Michael Cunningham is a U.S. novelist.
  Fact 1.2: Michael Cunningham is a U.S. screenwriter.

Retrieved items:
<item 1>
Isaac Asimov's Robot City
Isaac Asimov's Robot City is a series of novels written by various authors and loosely connected to Isaac Asimov's "Robot" series.

 It takes place between "The Robots of Dawn" and "Robots and Empire".

 Each volume is complete in itself, but they form a continuing series.

 The novels were written in response to a writing challenge issued by Asimov to write a series involving the Three Laws of Robotics, which brought about a
collaboration of several authors.

 Asimov provided outlines for stories which filled in the gap between Asimov's own robot stories and his "Foundation" series, explaining the disappearance of
the robots prior to the establishment of the galactic empire. "

Isaac Asimov's Robots and Aliens" followed in this series, with the same protagonists and many other characters.

 The common theme of all books of both series is the interaction between the characters and autonomous cities run and populated by robots (the "robot cities" of
the series title).

 Robot City was also released as a mystery game for the PC in 1995.

 The player takes the role of Derec.
</item 1>

<item 2>
Isaac Asimov
Isaac Asimov ( ; born Isaak Ozimov; January 2, 1920 – April 6, 1992) was a Russian-American writer and professor of biochemistry at Boston University.

 He was known for his works of science fiction and popular science.

 Asimov was a prolific writer, and wrote or edited more than 500 books and an estimated 90,000 letters and postcards. His books have been published in 9 of the
10 major categories of the Dewey Decimal Classification.
</item 2>

<item 3>
Isaac Asimov's Robot City: Perihelion
Isaac Asimov's Robot City: Perihelion is a book written in 1988 by William F. Wu.

 It is part of the series Isaac Asimov's Robot City, which was inspired by Isaac Asimov's "Robot" series.
</item 3>

<item 4>
Isaac Asimov's Magical Worlds of Fantasy
Isaac Asimov's Magical Worlds of Fantasy is a series of twelve themed paperback fantasy and science fiction anthologies edited by Isaac Asimov, Martin H.
Greenberg and Charles G. Waugh, a companion set to the ten volume "Isaac Asimov's Wonderful Worlds of Science Fiction", produced by the same editors.

 It was published by Signet/New American Library from 1983 to 1991.

 Volumes 1 and 2 were also issued in hardcover in an omnibus collection titled "Isaac Asimov's Magical Worlds of Fantasy: Witches & Wizards".
</item 4>

<item 5>
The Last Question
"The Last Question" is a science fiction short story by American writer Isaac Asimov.

 It first appeared in the November 1956 issue of "Science Fiction Quarterly" and was anthologized in the collections "Nine Tomorrows" (1959), "The Best of Isaac
Asimov" (1973), "Robot Dreams" (1986), the retrospective "Opus 100" (1969), and in "Isaac Asimov: The Complete Stories, Vol.

 1" (1990).

 It was Asimov's favorite short story of his own authorship, and is one of a loosely connected series of stories concerning a fictional computer called
Multivac.

 The story overlaps science fiction, theology, and philosophy.
</item 5>

<item 6>
Michael Cunningham
Michael Cunningham (born November 6, 1952) is a U.S. novelist and screenwriter.

 He is best known for his 1998 novel "The Hours", which won the Pulitzer Prize for Fiction and the PEN/Faulkner Award in 1999.

 Cunningham is a senior lecturer of creative writing at Yale University.
</item 6>

<item 7>
Orson Scott Card
Orson Scott Card (born August 24, 1951) is an American novelist, critic, public speaker, essayist, and columnist.

 He writes in several genres but is known best for science fiction.

 His novel "Ender's Game" (1985) and its sequel "Speaker for the Dead" (1986) both won Hugo and Nebula Awards, making Card the only author to win both science
fiction's top U.S. prizes in consecutive years.

 A feature film adaptation of "Ender's Game", which Card co-produced, was released in late October 2013 in Europe and on November 1, 2013, in North America.
</item 7>

<item 8>
Wandering Stars
Wandering Stars is an anthology of Jewish fantasy and science fiction, edited by American writer Jack Dann, originally published by Harper & Row in 1974.

 It represented, according to the book cover, "the first time in science fiction that the Jew - and the richness of his themes and particular points of view --
will appear without a mask."

 In his introduction, "Why Me?"

, Isaac Asimov discussed how many Jewish science fiction writers prior to that time had used gentile pen names in order to get published: "Many of the Jewish
pulp writers, however, used pen names as a matter of sound business.

 A story entitled "War Gods of the Oyster-Men of Deneb" did not carry conviction if it was written by someone named Chaim Itzkowitz."

 He then goes on to discuss the pen names of various Jewish writers included in this book.

 "Wandering Stars" is therefore of historical significance as the first science fiction anthology where Jewish writers openly identified themselves as such.

 It was followed by a second anthology, "More Wandering Stars", also edited by Jack Dann, published by Doubleday in 1981.
</item 8>

<item 9>
Isaac Asimov's Robot City: Odyssey
Isaac Asimov's Robot City: Odyssey is a young adult science fiction novel written in 1987 by Michael P. Kube-McDowell.

 It is part of the series Isaac Asimov's Robot City, inspired by Isaac Asimov's "Robot" series.

 The 1995 computer game Robot City is based on the plot of "Odyssey".
</item 9>

<item 10>
Isaac Asimov's Robots and Aliens
Isaac Asimov's Robots and Aliens is a series of novels written by various authors and loosely connected to Isaac Asimov's "Robot" series.

 Each volume is complete in itself, but they form a continuing series.

 The series follows the action of the novels of the "Isaac Asimov's Robot City" series, with the same protagonists Derec and Ariel, and many other characters.

 The plot deals with the Three Laws and encounters between robots and different varieties of alien life.
</item 10>
=== END ITEM id=8a4ac4853625f377 ===

=== ITEM id=e676163ebff68a32 ===
Gold sentences and facts:
Sentence 0: Jean-François André Clervoy (born 19 November 1958) is a French engineer and a CNES and ESA astronaut.
  Fact 0.0: Jean-François André Clervoy was born on 19 November 1958.
  Fact 0.1: Jean-François André Clervoy is a French engineer.
  Fact 0.2: Jean-François André Clervoy is a CNES astronaut.
  Fact 0.3: Jean-François André Clervoy is an ESA astronaut.
Sentence 1: Prof. Dr. Ulrich Hans Walter (born February 9, 1954) is a German physicist/engineer and a former DFVLR astronaut.
  Fact 1.0: Prof. Dr. Ulrich Hans Walter was born on February 9, 1954.
  Fact 1.1: Prof. Dr. Ulrich Hans Walter is a German physicist.
  Fact 1.2: Prof. Dr. Ulrich Hans Walter is a German engineer.
  Fact 1.3: Prof. Dr. Ulrich Hans Walter is a former DFVLR astronaut.

Retrieved items:
<item 1>
Jean-François Clervoy
Jean-François André Clervoy (born 19 November 1958) is a French engineer and a CNES and ESA astronaut.

 He is a veteran of three NASA Space Shuttle missions.
</item 1>

<item 2>
Ernst Messerschmid
Prof. Dr. Ernst Willi Messerschmid (born May 21, 1945) is a German physicist and former astronaut.
</item 2>

<item 3>
Patrick Baudry
Patrick Pierre Roger Baudry (born March 6, 1946 in Cameroon), is a retired Lieutenant Colonel in the French Air Force and a former CNES astronaut.

 In 1985, he became the second French citizen in space, after Jean-Loup Chrétien, when he flew aboard NASA's Space Shuttle mission STS-51-G.
</item 3>

<item 4>
Claude Nicollier
Claude Nicollier (born 2 September 1944 in Vevey, Switzerland) is the first astronaut from Switzerland.

 He has flown on four Space Shuttle missions.

 His first spaceflight (STS-46) was in 1992, and his final spaceflight (STS-103) was in 1999.

 He took part in two servicing missions to the Hubble Space Telescope (called STS-61 and STS-103).

 During his final spaceflight he participated in a spacewalk, becoming the first European Space Agency astronaut to do so during a Space Shuttle mission
(previous ESA astronauts conducted spacewalks aboard "Mir", see List of spacewalks and moonwalks 1965–1999).

 In 2000 he was assigned to the Astronaut Office Extravehicular Activity Branch, while maintaining a position as Lead ESA Astronaut in Houston.

 Nicollier retired from ESA in April 2007.
</item 4>

<item 5>
Ulrich Walter
Prof. Dr. Ulrich Hans Walter (born February 9, 1954) is a German physicist/engineer and a former DFVLR astronaut.
</item 5>

<item 6>
Jean-Loup Chrétien
Jean-Loup Jacques Marie Chrétien (born 20 August 1938) is a French retired "Général de Brigade" (brigadier general) in the "Armée de l'Air" (French air force),
and a former CNES spationaut.

 He flew on two Franco-Soviet space missions and a NASA Space Shuttle mission.

 Chrétien was the first Frenchman and the first western European in space.
</item 6>

<item 7>
Hans Schlegel
Hans Wilhelm Schlegel (Überlingen, 3 August 1951) is a German physicist, an ESA astronaut, and a veteran of two NASA Space Shuttle missions.
</item 7>

<item 8>
Ulf Merbold
Dr. Ulf Dietrich Merbold (born June 20, 1941) is the first West German citizen and second German native (after Sigmund Jähn) to have flown in space.

 He is also the first member of the European Space Agency Astronaut Corps to participate in a spaceflight mission and the first non-US citizen to reach orbit in
a US spacecraft.

 In 1983, he and Byron Lichtenberg became the first Payload Specialists to fly on the shuttle.
</item 8>

<item 9>
Otto Struve
Otto Struve (August 12, 1897 – April 6, 1963) was a astronomer.

 In Russian, his name is sometimes given as Otto Lyudvigovich Struve (Отто Людвигович Струве); however, he spent most of his life and his entire scientific
career in the United States.

 Otto was the descendant of famous astronomers of the Struve family; he was the son of Ludwig Struve, grandson of Otto Wilhelm von Struve and great-grandson of
Friedrich Georg Wilhelm von Struve.

 He was also the nephew of Karl Hermann Struve.
</item 9>

<item 10>
Walter Cunningham
Ronnie Walter Cunningham (born March 16, 1932), (Col, USMCR, Ret.)

, better known as Walter Cunningham, is a retired American astronaut.

 In 1968, he was a Lunar Module Pilot on the Apollo 7 mission.

 He was NASA's third civilian astronaut (after Neil Armstrong and Elliot See), and has also been a fighter pilot, physicist, entrepreneur, venture capitalist,
author of "The All-American Boys", lecturer, and host of the radio show "Lift-off to Logic".

 Walter Cunningham attended UCLA.
</item 10>
=== END ITEM id=e676163ebff68a32 ===

=== ITEM id=6c0f3401a453abe3 ===
Gold sentences and facts:
Sentence 0: Kenneth William "Ken" Kwapis (born August 17, 1957) is an American film and television director and screenwriter.
  Fact 0.0: Kenneth William Kwapis, also known as Ken Kwapis, was born on August 17, 1957.
  Fact 0.1: Ken Kwapis is an American film director.
  Fact 0.2: Ken Kwapis is an American television director.
  Fact 0.3: Ken Kwapis is an American screenwriter.
Sentence 1:  He specialized in the single-camera sitcom in the 1990s and 2000s and has directed feature films such as "Sesame Street Presents Follow That Bird" (1985), "The Sisterhood of the Traveling Pants" (2005), and "He's Just Not That into You" (2009).
  Fact 1.0: He specialized in the single-camera sitcom in the 1990s.
  Fact 1.1: He specialized in the single-camera sitcom in the 2000s.
  Fact 1.2: He has directed feature films.
  Fact 1.3: He directed Sesame Street Presents Follow That Bird, released in 1985.
  Fact 1.4: He directed The Sisterhood of the Traveling Pants, released in 2005.
  Fact 1.5: He directed He's Just Not That into You, released in 2009.
Sentence 2:  As a filmmaker, he has directed films such as Valle de Lágrimas (2006) and Peace Grows (2004).
  Fact 2.0: As a filmmaker, he has directed Valle de Lágrimas, released in 2006.
  Fact 2.1: As a filmmaker, he has directed Peace Grows, released in 2004.

Retrieved items:
<item 1>
Chris Gorak
Chris Gorak is an American film director who started as an art director and production designer.

 He directed the 2006 film "Right at Your Door" and the 2011 film "The Darkest Hour".

 Gorak began working in the film industry in the 1990s as an art director.

 In the 2000s, he worked on several films as production designer before becoming a film director.
</item 1>

<item 2>
Jon Paul Puno
Jon Paul Puno (born 1985) is an American filmmaker and classical crossover singer.

 As a filmmaker, he has directed films such as Valle de Lágrimas (2006) and Peace Grows (2004).

 And also played a small role in the 2006 movie I Will Always Love You (film).
</item 2>

<item 3>
Dunston Checks In
Dunston Checks In is a 1996 American family comedy film starring Eric Lloyd, Graham Sack, Jason Alexander, Faye Dunaway, Rupert Everett, Paul Reubens, Glenn
Shadix, and introducing Sam the Orangutan as Dunston.

 It was written by John Hopkins and Bruce Graham and directed by Ken Kwapis.
</item 3>

<item 4>
Guylaine Maroist
Guylaine Maroist is a Canadian journalist, filmmaker, musician, scriptwriter and film director.

 She is well known for her documentary productions such as "Gentilly or Not To Be", "Time Bombs", "The Disunited States of Canada", and "God Save Justin
Trudeau".

 In 2011 she received the Governor General’s History Award for Popular History (The Pierre Berton Award) for her TV documentary series "J’ai la mémoire qui
tourne" ("My Memories On a Roll").

 She is President of Productions de la Ruelle, a documentary film production company in Montreal, and President of Les Artistes pour la Paix, a Quebec NGO
advocating peace and nuclear disarmament.
</item 4>

<item 5>
Paul Schrader
Paul Joseph Schrader (born July 22, 1946) is an American screenwriter, film director, and film critic.

 Schrader wrote or co-wrote screenplays for four Martin Scorsese films: "Taxi Driver" (1976), "Raging Bull" (1980), "The Last Temptation of Christ" (1988), and
"Bringing Out the Dead" (1999).

 Schrader has also directed 18 feature films, including his directing debut crime drama, "Blue Collar" (co-written with his brother, Leonard Schrader), the
crime drama "Hardcore" (a loosely autobiographical film also written by Schrader), his 1982 remake of the horror classic "Cat People", the crime drama "American
Gigolo" (1980), the biographical drama "" (1985), the cult film "Light Sleeper" (1992), the drama "Affliction" (1997), the biographical film "Auto Focus"
(2002), and the erotic dramatic thriller "The Canyons" (2013).
</item 5>

<item 6>
Dana Brown
Dana Brown (born December 11, 1959 in Dana Point, California) is an American surfer and filmmaker, and is the oldest son of filmmaker Bruce Brown.

 His films include "The Endless Summer Revisited" (2000) which is made up of unused footage from "The Endless Summer" (1964) and "The Endless Summer II" (1994),
as well as some original interviews with the stars of those films.

 His first all-original film was "Step Into Liquid" (2003) followed by a documentary on the Baja 1000 titled "Dust to Glory" (2005).

 In 2009, he debuted a new film called "Highwater" during the 100th anniversary of the Santa Monica Pier; the film follows life on the North Shore and the
surfers who compete in the Vans Triple Crown of Surfing.

 In 2014, the movie "On Any Sunday, The Next Chapter" continues the saga of motocross documentaries which began with the 1972 Academy Award for Documentary
Feature nominated film "On Any Sunday" (1971).
</item 6>

<item 7>
John Bailey (cinematographer)
John Ira Bailey, A.S.C. (born August 10, 1942) is an American cinematographer and film director best known for his collaborations with directors Paul Schrader,
Lawrence Kasdan, Michael Apted, and Ken Kwapis.

 In August 2017, Bailey was elected President of the Academy of Motion Picture Arts and Sciences for a four-year term.
</item 7>

<item 8>
Kevin Brownlow
Kevin Brownlow (born 2 June 1938) is a British film historian, television documentary-maker, filmmaker, author, and film editor.

 Brownlow is best known for his work documenting the history of the silent era.

 Brownlow became interested in silent film at the age of eleven.

 This interest grew into a career spent documenting and restoring film.

 He has rescued many silent films and their history.

 His initiative in interviewing many largely forgotten, elderly film pioneers in the 1960s and 1970s preserved a legacy of early mass-entertainment cinema.

 Brownlow received an Academy Honorary Award at the 2nd Annual Governors Awards given by the Academy of Motion Picture Arts and Sciences on 13 November 2010.

 This was the first occasion on which an Academy Honorary Award was given to a film preservationist.
</item 8>

<item 9>
Ken Kwapis
Kenneth William "Ken" Kwapis (born August 17, 1957) is an American film and television director and screenwriter.

 He specialized in the single-camera sitcom in the 1990s and 2000s and has directed feature films such as "Sesame Street Presents Follow That Bird" (1985), "The
Sisterhood of the Traveling Pants" (2005), and "He's Just Not That into You" (2009).
</item 9>

<item 10>
Sexual Life
Sexual Life is a 2005 comedy-drama, independent film written and directed by Ken Kwapis, who would go on to chronicle modern romantic life in the better-known
"He's Just Not That Into You" in 2009.

 Produced by Ken Aguado and distributed by Showtime Independent Films.

 Cast members include Azura Skye, Carla Gallo, Anne Heche, Elizabeth Banks, Tom Everett Scott, and Steven Weber.
</item 10>
=== END ITEM id=6c0f3401a453abe3 ===

=== ITEM id=86fa3b1662993f0d ===
Gold sentences and facts:
Sentence 0: War/Dance is a 2007 American documentary film written and directed by Sean Fine and Andrea Nix Fine and produced by Shine Global's Susan MacLaury, a professor at Kean University, and Albie Hecht.
  Fact 0.0: War/Dance is a 2007 American documentary film.
  Fact 0.1: War/Dance was written by Sean Fine and Andrea Nix Fine.
  Fact 0.2: War/Dance was directed by Sean Fine and Andrea Nix Fine.
  Fact 0.3: War/Dance was produced by Shine Global's Susan MacLaury.
  Fact 0.4: Susan MacLaury is a professor at Kean University.
  Fact 0.5: War/Dance was produced by Albie Hecht.
Sentence 1: The Sorrow and the Pity (French: Le Chagrin et la Pitié ) is a two-part 1969 documentary film by Marcel Ophüls about the collaboration between the Vichy government and Nazi Germany during World War II.
  Fact 1.0: The Sorrow and the Pity is a documentary film.
  Fact 1.1: The Sorrow and the Pity's French title is Le Chagrin et la Pitié.
  Fact 1.2: The Sorrow and the Pity is in two parts.
  Fact 1.3: The Sorrow and the Pity was released in 1969.
  Fact 1.4: The Sorrow and the Pity was made by Marcel Ophüls.
  Fact 1.5: The Sorrow and the Pity is about the collaboration between the Vichy government and Nazi Germany.
  Fact 1.6: That collaboration occurred during World War II.

Retrieved items:
<item 1>
The Sorrow and the Pity
The Sorrow and the Pity (French: Le Chagrin et la Pitié ) is a two-part 1969 documentary film by Marcel Ophüls about the collaboration between the Vichy
government and Nazi Germany during World War II.

 The film uses interviews with a German officer, collaborators, and resistance fighters from Clermont-Ferrand.

 They comment on the nature of and reasons for collaboration.

 The reasons include antisemitism, anglophobia, fear of Bolsheviks and Soviet invasion, the desire for power, and simple caution.
</item 1>

<item 2>
The entries survey documentaries, films, TV, and film-related concepts: Los Angeles hip-hop, Aboriginal country music, Latin American politics, oceans, nerd
culture, Iraqi life and war, Bougainville, education, disability, poetry, surfing, and family relationships. Featured works include *Gasland*, *Baghdad ER*,
*War/Dance*, *First Life*, and *The Coconut Revolution*. Several received major awards or nominations. The material also profiles filmmaker Wayne Coles-Janess
and defines
</item 2>

<item 3>
Tender Fictions
Tender Fictions is a 1996 autobiographical documentary film directed by American experimental filmmaker Barbara Hammer.

 It is the second of a trilogy of documentary films that includes "Nitrate Kisses" and "History Lessons".

 Together, the three films are sometimes known as the "History trilogy".

 "Tender Fictions" details Hammer's life and her attempts to "construct" a self.

 The film was nominated for a prize at the 1996 Sundance Film Festival.
</item 3>

<item 4>
The entries cover documentaries on Boston hardcore (All Ages, 2012), punk (Punk’s Not Dead, 2007; American Hardcore, 2006), bipolar disorder, artistic aging,
slavery, archaeology, bookstores, and Charles Bukowski; plus Barbara Hammer’s autobiographical Tender Fictions. Fiction films include Bakshi’s acclaimed X-rated
Heavy Traffic, Heavyweights, several American Pie and Bring It On sequels, Schrader’s Hardcore, Blue Collar, and The
</item 4>

<item 5>
Marcel Ophüls
Marcel Ophuls (] ; born 1 November 1927) is a documentary film maker and former actor, best known for his films "The Sorrow and the Pity" and "".
</item 5>

<item 6>
In the Mirror of Maya Deren
The documentary In the Mirror of Maya Deren ("Im Spiegel der Maya Deren", 2002) is a film about avant garde filmmaker Maya Deren (1917-1961) by Austrian film
maker Martina Kudlacek.

 It is based on the biography "The Legend of Maya Deren".
</item 6>

<item 7>
War/Dance
War/Dance is a 2007 American documentary film written and directed by Sean Fine and Andrea Nix Fine and produced by Shine Global's Susan MacLaury, a professor
at Kean University, and Albie Hecht.

 It was nominated for the 2008 Academy Award for Best Documentary Feature and received the Emmy Award for Best Documentary and Best Cinematography in 2010.
</item 7>

<item 8>
Polio v/s Polio victims
Polio v/s Polio Victims is the National Award Winning Documentary film.

 The documentary won "Best Motivational/ Educational Film" at the 56th National Film Awards.

 The documentary follows a group of polio victims who took to the streets in 2008 to spread the message of the Pulse Polio Campaign.

 The film follows them as they go door to door, in various slums in Mumbai, to spread awareness about the upcoming "Pulse Polio Day".
</item 8>

<item 9>
In the Shadow of the Palms
In the Shadow of the Palms (2005) is a documentary produced and directed by the Australian filmmaker Wayne Coles-Janess.

 He filmed it in Iraq prior to, during and after the fall of Saddam Hussein after the United States invasion of 2003.

 An Arab-language film, it documents the changes in Iraqi society and the lives of ordinary Iraqis by focusing on a cross-section of individuals.
</item 9>

<item 10>
The Great War (documentary)
The Great War is a 26-episode documentary series from 1964 on the First World War.

 The documentary was a co-production of the Imperial War Museum, the British Broadcasting Corporation, the Canadian Broadcasting Corporation and the Australian
Broadcasting Commission.

 The narrator was Michael Redgrave, with readings by Marius Goring, Ralph Richardson, Cyril Luckham, Sebastian Shaw and Emlyn Williams.

 Each episode is  40 minutes long.
</item 10>
=== END ITEM id=86fa3b1662993f0d ===

=== ITEM id=9693a2f8abcc8f42 ===
Gold sentences and facts:
Sentence 0: Cherie Ann Currie (born November 30, 1959) is an American musician (instrumentalist, singer, songwriter), actress and artist.
  Fact 0.0: Cherie Ann Currie was born November 30, 1959.
  Fact 0.1: Cherie Ann Currie is an American musician.
  Fact 0.2: Cherie Ann Currie is an instrumentalist, singer, and songwriter.
  Fact 0.3: Cherie Ann Currie is an actress.
  Fact 0.4: Cherie Ann Currie is an artist.
Sentence 1: Steve Augeri (born January 30, 1959) is an American rock singer best known for his work with Tall Stories, Tyketto, and Journey.
  Fact 1.0: Steve Augeri was born January 30, 1959.
  Fact 1.1: Steve Augeri is an American rock singer.
  Fact 1.2: Steve Augeri is best known for his work with Tall Stories.
  Fact 1.3: Steve Augeri is best known for his work with Tyketto.
  Fact 1.4: Steve Augeri is best known for his work with Journey.

Retrieved items:
<item 1>
Steve Augeri
Steve Augeri (born January 30, 1959) is an American rock singer best known for his work with Tall Stories, Tyketto, and Journey.
</item 1>

<item 2>
Steve Augeri is an American rock singer known for Tall Stories, Tyketto, and Journey. English multi-instrumentalist Steve Winwood performed with the Spencer
Davis Group, Traffic, and Blind Faith; his debut solo album reached UK No. 12, while two compilations document his work. His 1986 hit “Back in the High Life
Again” topped the US Adult Contemporary chart for three weeks. Pedal-steel guitarist Speedy West played with Jimmy Bryant and on recordings by Tennessee
</item 2>

<item 3>
Messin' with the Boys
Messin' with the Boys is the second post-Runaways album by Cherie Currie, released in 1980 For this album Currie worked with her identical twin sister, Marie
Currie.

 Their band was called "Cherie and Marie Currie".

 Marie was a guest vocalist on Cherie's first album, so Marie went on tour with Cherie to support her first album, "Beauty's Only Skin Deep".

 When Marie would join Cherie on stage to sing the encores the audience would go wild.

 So Cherie ran with the idea of two blonds are better than one.

 The idea paid off because "Messin' with the Boys" received more radio play than "Beauty's Only Skin Deep", and their song "Since You Been Gone" made it to 95
on US charts.

 The single "This Time" and the album Messin' with the Boys made the top 200 on U.S. charts.

 This makes Messin' with the Boys Cherie Currie's most successful album.

 "I Just Love the Feeling" originally surfaced on the 1974 album, "S.S.
</item 3>

<item 4>
Reverie (Cherie Currie album)
Reverie is the third full-length studio album by Cherie Currie.

 Released on iTunes March 16, 2015.

 Cherie released the CD version of this album June 5, 2015 on her eBay page cheriecurriedirect.

 There is a 35-year gap between Cherie's last full-length studio album, 1980's Messin' with the Boys (with Marie Currie), and 2015's Reverie.

 This is last studio album Kim Fowley produced before his death.

 Kim helped Cherie release this album to make amends with her after all the money he swindled her out of when she was in the Runaways and for releasing her and
Marie's music on Young and Wild without their approval.

 After Kim's death Cherie's son, Jake Hays, took over producing.
</item 4>

<item 5>
CeCe Peniston’s 1998 “Nobody Else,” intended to lead a canceled album, never appeared on its planned compilation; she later released two Silk singles. Cherie
Currie’s 2015 *Reverie* ended a 35-year album gap and was Kim Fowley’s final production; her son finished it after Fowley died. *Young and Wild* compiles Currie
sisters’ recordings and a new song. Their 1980 *Messin’ with the
</item 5>

<item 6>
Cher’s acting work has focused mainly on comedy, drama, and romance. She appeared in 13 films, including two cameos, had one starring theater role, and directed
a segment of *If These Walls Could Talk* (1996) and several Geffen-era music videos. She also appeared in international TV commercials and Lori Davis print
advertising (1992). Before her film career, she had hit records in the 1960s as a solo artist and with Sonny Bono as
</item 6>

<item 7>
Young and Wild (album)
Young and Wild is a compilation by Cherie & Marie Currie.

 This album has all 10 original tracks from Messin' with the Boys, six songs from Beauty's Only Skin Deep, three songs Cherie Currie sang with The Runaways, and
one new track co-written by Marie Currie, "Longer Than Forever".

 "Longer Than Forever" was the B side of the single "Since You Been Gone".
</item 7>

<item 8>
Cherie Currie
Cherie Ann Currie (born November 30, 1959) is an American musician (instrumentalist, singer, songwriter), actress and artist.

 Currie was the lead vocalist of the Runaways, a rock band from Los Angeles, in the mid-to-late 1970s.

 After the Runaways, she became a solo artist.

 Then she teamed up with her identical twin sister, Marie Currie, and released an album with her.

 They released a duet "Since You Been Gone" which charted number 95 on US charts.

 Their band was called Cherie and Marie Currie.

 She is also well known for her role in the movie "Foxes".
</item 8>

<item 9>
Cherushii
Cherushii (Chelsea Faith Dolan) (September 14, 1983 – December 2, 2016) was an American electronic music producer, DJ, live performer, and radio host based in
San Francisco.

 Influenced by house music, Detroit techno, and disco, she released several recordings on the independent dance music label 100% Silk, including "Queen of
Cups", "Far Away So Close", and "Memory of Water", reissued in 2017.

 KQED named Cherushii's album "Meow Wolf’s Arcade Soundtracks: Wiggy’s Plasma Plex" one of the 10 best bay area albums of 2016.

 While Cherushii has received little recognition from mainstream media, her music is of enormous influence in the underground music scene.

 Fellow electronic musician Maria Minerva has called Cherushii criminally underrated.
</item 9>

<item 10>
Marie Currie
Marie Michelle Currie (born November 30, 1959) is an American singer, songwriter, actress, and artist.

 Currie is best known for playing in a band with her identical twin Cherie Currie called Cherie & Marie Currie.

 Their song "Since You Been Gone" charted at number 95 on the US charts.

 Marie played Singing Maid Marie in "The Rosebud Beach Hotel" and is now a popular multi-media sculptress and artist.
</item 10>
=== END ITEM id=9693a2f8abcc8f42 ===
