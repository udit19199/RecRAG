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

=== ITEM id=d24ce72b391990c5 ===
Gold sentences and facts:
Sentence 0: Eat to Live: The Amazing Nutrient-Rich Program for Fast and Sustained Weight Loss is a book written in 2003 by Dr. Joel Fuhrman, which he revised in 2011.
  Fact 0.0: Eat to Live is a book.
  Fact 0.1: The full title of Eat to Live is 'Eat to Live: The Amazing Nutrient-Rich Program for Fast and Sustained Weight Loss'.
  Fact 0.2: Eat to Live was written in 2003.
  Fact 0.3: Eat to Live was written by Dr. Joel Fuhrman.
  Fact 0.4: Dr. Joel Fuhrman revised Eat to Live in 2011.
Sentence 1: Joel Fuhrman (born December 2, 1953) is an American celebrity doctor who advocates what he calls a micronutrient-rich diet.
  Fact 1.0: Joel Fuhrman was born on December 2, 1953.
  Fact 1.1: Joel Fuhrman is American.
  Fact 1.2: Joel Fuhrman is a celebrity doctor.
  Fact 1.3: Joel Fuhrman advocates a diet that he calls a micronutrient-rich diet.

Retrieved items:
<item 1>
Eat Smart, Move More, Weigh Less
Eat Smart, Move More, Weigh Less is a 15-week adult weight management program that uses strategies based on evidence for weight loss and/or weight maintenance.

 The Eat Smart Move More Weigh Less classes focus on the 12 evidence-based eating and physical activity behaviors for weight management.

 The program does not provide a prescriptive diet plan, it teaches small lifetime changes.

 The program teaches mindfulness as a strategy to become more aware of eating and physical activity.

 Eat Smart, Move More, Weigh Less includes methods for planning and tracking healthy eating and physical activity behaviors.

 In 2007, Eat Smart, Move More, Weigh Less curriculum was developed by a writing team from NC State University and NC Division of Public Health.

 A complete listing of authors and their respective affiliations can be found on the program website.

 The 15-lesson curriculum was peer reviewed by state and local nutrition and physical activity professionals, representatives from the medical community, and a
CDC project officer.
</item 1>

<item 2>
Diet in Hinduism
Diet in Hinduism varies with its diverse traditions.

 The ancient and medieval Hindu texts do not explicitly prohibit eating meat, but they do strongly recommend ahimsa—non-violence against all life forms
including animals.

 Many Hindus prefer a vegetarian or lacto-vegetarian lifestyle, and methods of food production that are in sync with nature, compassionate, and respectful of
other life forms as well as nature.
</item 2>

<item 3>
Joel Fuhrman
Joel Fuhrman (born December 2, 1953) is an American celebrity doctor who advocates what he calls a micronutrient-rich diet.
</item 3>

<item 4>
Scarsdale diet
The Scarsdale diet is a fad diet designed for weight loss created in the 1970s by Herman Tarnower, named for the town in New York where he practiced cardiology,
described in the book "The Complete Scarsdale Medical Diet plus Dr. Tarnower's Lifetime Keep-Slim Program", which Tarnower wrote with an author of self-help
books, Sam Sinclair Baker.

 The diet carries potential health risks and does not instill the kind of healthy eating habits required for sustainable weight loss.
</item 4>

<item 5>
Management of obesity
Management of obesity can include lifestyle changes, medications, or surgery.

 The main treatment for obesity consists of dieting and physical exercise.

 Diet programs may produce weight loss over the short term, but maintaining this weight loss is frequently difficult and often requires making exercise and a
lower calorie diet a permanent part of an individual's lifestyle.

 Success rates of long-term weight loss maintenance with lifestyle changes are low, ranging from 2 to 20%.

 Dietary and lifestyle changes are effective in limiting excessive weight gain in pregnancy and improve outcomes for both the mother and the child.

 The National Institutes of Health recommend a weight loss goal of 5% to 10% of the person's current weight over six months.
</item 5>

<item 6>
Very-low-calorie diet
Very low calorie diet (VLCD) is a diet with very or extremely low daily food energy consumption.

 It is defined as a diet of 800 kcal per day or less.

 VLCDs are formulated, nutritionally complete, liquid meals containing 800 kilocalories or less per day.

 VLCDs also contain the recommended daily requirements for vitamins, minerals, trace elements, fatty acids and protein.

 Carbohydrate may be entirely absent, or substituted for a portion of the protein; this choice has important metabolic effects.

 The VLCD products are usually a powder which is mixed with water or another low-food-energy liquid.

 The VLCD is prescribed on a case to case basis for rapid weight loss (about 1.5 to 2.5 kilograms or 3 to 5 pounds per week) in people with Body Mass Index of
30 and above.

 The health care provider can recommend the diet to a patient with a BMI between 27 and 30 if the medical complications the patient has due to being overweight
present serious health risks.

 It results in 4% more weight loss over the short term as compared to control.
</item 6>

<item 7>
Diet food
Diet food (or dietetic food) refers to any food or beverage whose recipe is altered to reduce fat, carbohydrates, and/or sugar in order to make it part of a
weight loss program or diet.

 Such foods are usually intended to assist in weight loss or a change in body type, although bodybuilding supplements are designed to aid in gaining weight or
muscle.
</item 7>

<item 8>
Weight Wins
Weight Wins is a UK weight loss business which has developed and commercialised the concept of personal contracts for weight loss.

 The company has attracted attention from UK and international media for its ‘Pounds for Pounds’ weight loss programme and for operating the first trial by the
UK’s National Health Service (NHS) of a weight loss incentive programme.
</item 8>

<item 9>
Judy Mazel
Judy Mazel (December 20, 1943 – October 12, 2007) was the American weight loss advocate and author of the ""The Beverly Hills Diet"", which became a 1981 best
seller and a nationwide diet craze.

 Judy Mazel was a devout practicing Jew and was raised in a conservative Ashkenazi Jewish home.
</item 9>

<item 10>
Eat to Live
Eat to Live: The Amazing Nutrient-Rich Program for Fast and Sustained Weight Loss is a book written in 2003 by Dr. Joel Fuhrman, which he revised in 2011.

 The book offers a formula: health = nutrients / calories.
</item 10>
=== END ITEM id=d24ce72b391990c5 ===

=== ITEM id=167627aba0c91d08 ===
Gold sentences and facts:
Sentence 0:  The early years of the company were dedicated to issuing recordings by jazz pianist Dave Brubeck, who was also one of its investors, but the label is known more for its recordings of comedian Lenny Bruce, jazz musician Vince Guaraldi, and the rock band Creedence Clearwater Revival.
  Fact 0.0: The early years of the company were dedicated to issuing recordings by jazz pianist Dave Brubeck.
  Fact 0.1: Dave Brubeck was also one of the company's investors.
  Fact 0.2: The label is known more for its recordings of comedian Lenny Bruce.
  Fact 0.3: The label is known more for its recordings of jazz musician Vince Guaraldi.
  Fact 0.4: The label is known more for its recordings of the rock band Creedence Clearwater Revival.
Sentence 1: Vincent Anthony Guaraldi (July 17, 1928 – February 6, 1976), born Vincent Anthony Dellaglio, was an American jazz pianist noted for his innovative compositions and arrangements and for composing music for animated television adaptations of the "Peanuts" comic strip, as well as his performances on piano as a member of Cal Tjader's late 50s ensemble and his own solo career which included the radio hit "Cast Your Fate to the Wind".
  Fact 1.0: Vincent Anthony Guaraldi was born on July 17, 1928.
  Fact 1.1: Vincent Anthony Guaraldi died on February 6, 1976.
  Fact 1.2: Vincent Anthony Guaraldi was born Vincent Anthony Dellaglio.
  Fact 1.3: Vincent Anthony Guaraldi was an American jazz pianist.
  Fact 1.4: Vincent Anthony Guaraldi was noted for his innovative compositions and arrangements.
  Fact 1.5: Vincent Anthony Guaraldi was noted for composing music for animated television adaptations of the 'Peanuts' comic strip.
  Fact 1.6: Vincent Anthony Guaraldi performed on piano as a member of Cal Tjader's late 50s ensemble.
  Fact 1.7: Vincent Anthony Guaraldi had a solo career.
  Fact 1.8: Vincent Anthony Guaraldi's solo career included the radio hit 'Cast Your Fate to the Wind'.

Retrieved items:
<item 1>
Doug Clifford
Douglas "Cosmo" Clifford (born April 24, 1945 in Palo Alto, California) is an American drummer, best known as a founding member of Creedence Clearwater Revival.

 After the group dissolved in the early 1970s, Clifford released a solo album and later joined CCR bassist Stu Cook in the Don Harrison Band.

 In 1995, Clifford and Cook formed the band Creedence Clearwater Revisited, performing live versions of Creedence Clearwater Revival songs.
</item 1>

<item 2>
The Best of Creedence Clearwater Revival
The Best of Creedence Clearwater Revival is a compilation album by American rock band Creedence Clearwater Revival, released in 1977.

 The album features all the tracks of Chronicle which was released a year earlier in 1976, as well as "Good Golly Miss Molly", "Born on the Bayou", "Cotton
Fields", "Hello Mary Lou", "The Midnight Special", "Walk on the Water", and "Night Time Is the Right Time" which would appear on the 1986 release "Chronicle,
Vol.

 2", and "Bootleg" which does not appear on either volume.
</item 2>

<item 3>
Creedence Clearwater Revival: Box Set
Creedence Clearwater Revival: Box Set is a box set by Creedence Clearwater Revival, released in 2001.

 It contains all of their complete studio albums, two complete live albums, and material recorded by the band under their previous names "The Golliwogs" and
"The Blue Velvets".

 In November 2013 the box set was reissued with different artwork.

 This box set spans their career, and contains all of their complete materials.
</item 3>

<item 4>
Creedence Clearwater Revival (album)
Creedence Clearwater Revival is the debut studio album by the American rock band Creedence Clearwater Revival, released in 1968.
</item 4>

<item 5>
The Long Road Home
The Long Road Home: The Ultimate John Fogerty/Creedence Collection is a compilation album by American roots rock singer-songwriter John Fogerty, released on
November 1, 2005, by Fantasy Records.

 It compiles songs from Fogerty's solo career and his band Creedence Clearwater Revival.

 The title refers to Fogerty's return to Fantasy Records, after a lengthy stint with Warner Bros.

 Records and a brief stint with DreamWorks Records.
</item 5>

<item 6>
Vince Guaraldi
Vincent Anthony Guaraldi (July 17, 1928 – February 6, 1976), born Vincent Anthony Dellaglio, was an American jazz pianist noted for his innovative compositions
and arrangements and for composing music for animated television adaptations of the "Peanuts" comic strip, as well as his performances on piano as a member of
Cal Tjader's late 50s ensemble and his own solo career which included the radio hit "Cast Your Fate to the Wind".
</item 6>

<item 7>
Jerry Beck
Jerry Beck (born February 9, 1955 in New York City) is an American animation historian, author, blogger, and video producer.

 The author and/or editor of several books on classic American animation and classic characters, including "The 50 Greatest Cartoons" (1994), "The Animated
Movie Guide" (2005), "Not Just Cartoons: Nicktoons!"

 (2007), "The Flintstones: The Official Guide to the Cartoon Classic" (2011), "The Hanna-Barbera Treasury: Rare Art Mementos from Your Favorite Cartoon
Classics" (2007), "The SpongeBob SquarePants Experience: A Deep Dive into the World of Bikini Bottom" (2013), "Pink Panther: The Ultimate Guide" (2005), and
"Looney Tunes and Merrie Melodies: A Complete Illustrated Guide to the Warner Bros.

 Cartoons" (with Will Friedwald, 1989) alongside "The 100 Greatest Looney Tunes Cartoons" (2010).

 He is also an authority on the making of modern films, with his books detailing the art of "Mr.
</item 7>

<item 8>
Creedence Clearwater Revisited
Creedence Clearwater Revisited is an American rock band formed in 1995 by two former members of Creedence Clearwater Revival – a much more well-known band with
a very similar name.

 It is known for playing the music that was originally made famous by Creedence Clearwater Revival.

 The two common band members are Stu Cook (bass) and Doug "Cosmo" Clifford (drums).
</item 8>

<item 9>
Fantasy Records
Fantasy Records is an American record company and label founded by brothers Max and Sol Weiss in 1949.

 The early years of the company were dedicated to issuing recordings by jazz pianist Dave Brubeck, who was also one of its investors, but the label is known
more for its recordings of comedian Lenny Bruce, jazz musician Vince Guaraldi, and the rock band Creedence Clearwater Revival.
</item 9>

<item 10>
John Fogerty (album)
John Fogerty is the second solo studio album by former Creedence Clearwater Revival vocalist/guitarist John Fogerty, released in 1975.

 It was released by Asylum Records in the United States and Fantasy Records internationally.

 As with the Creedence Clearwater Revival records, the album consists of a mix of originals and cover songs.

 Although the album is eponymously titled, Fogerty himself refers to it as "Old Shep"; Shep was the name of his dog, who appears on the cover with him.
</item 10>
=== END ITEM id=167627aba0c91d08 ===

=== ITEM id=d8d97df2d88ebd76 ===
Gold sentences and facts:
Sentence 0: Adam Fox (1883 – 1977), Canon, was the Dean of Divinity at Magdalen College, Oxford.
  Fact 0.0: Adam Fox lived from 1883 to 1977.
  Fact 0.1: Adam Fox was a Canon.
  Fact 0.2: Adam Fox was the Dean of Divinity at Magdalen College, Oxford.
Sentence 1:  He was one of the first members of the "Inklings", a literary group which also included C.S. Lewis and J.R.R. Tolkien.
  Fact 1.0: Adam Fox was one of the first members of the Inklings.
  Fact 1.1: The Inklings was a literary group.
  Fact 1.2: The Inklings also included C.S. Lewis.
  Fact 1.3: The Inklings also included J.R.R. Tolkien.
Sentence 2: The Inklings was an informal literary discussion group associated with the University of Oxford, England, for nearly two decades between the early 1930s and late 1949.
  Fact 2.0: The Inklings was an informal literary discussion group.
  Fact 2.1: The Inklings was associated with the University of Oxford, England.
  Fact 2.2: The Inklings lasted for nearly two decades.
  Fact 2.3: The Inklings operated between the early 1930s and late 1949.

Retrieved items:
<item 1>
James F. Light
James F. Light was an American literary scholar, university vice president, and provost.

 During his academic career, he helped revive the works of satirist Nathanael West, with the first book length critical study of his work, "Nathanael West: An
Interpretive Study", (Northwestern Univ.

 Press, 1961).

 He was also the leading authority on John William De Forest, the early American realist whose work he critiqued in "John William De Forest" (Twayne Pub.,
1965), and he wrote extensively on J. D. Salinger, Robert Penn Warren and others.
</item 1>

<item 2>
Donald Allen
Donald Merriam Allen (Iowa, 1912 – San Francisco, August 29, 2004) was an influential editor, publisher, and translator of contemporary American literature.

 He is perhaps best known for his project "The New American Poetry 1945-1960" (1960), among the several important anthologies of contemporary American
innovative writing he made available to the public.

 Allen began his working life as a Japanese translator within the US military, serving in WWII.

 After his military service, Allen became an editor at Grove Press, where he worked for sixteen years.

 He was one of the first translators of the Romanian-French Absurdist playwright Eugène Ionesco, and Allen's 1958 volume "Four Plays of Eugène Ionesco" helped
to introduce the playwright to American audiences in the 1960s.
</item 2>

<item 3>
Andrew Benjamin
Andrew Benjamin (born 1952, Australia) is an Australian philosopher.

 Currently he holds two academic posts: Distinguished Professor of Philosophy and the Humanities at Kingston University, London and Professor of Philosophy and
Jewish Thought in the Department of Philosophy and the Australian Centre for Jewish Civilization at Monash University, Melbourne, Australia.

 Benjamin first came to critical attention with his writings in continental philosophy, writing articles and editing books on the thinking of Martin Heidegger,
Theodor Adorno, Walter Benjamin, Julia Kristeva and Jean-François Lyotard.

 Benjamin has become involved in the field of architecture, to the extent that he has also taught in various schools of architecture in UK, US and Australia.
</item 3>

<item 4>
Falmouth University
Falmouth University (Cornish: Pennskol Aberfala ) is a specialist University for the creative industries based in Falmouth and Penryn, Cornwall, England.

 Founded in 1902, it had previously been the Falmouth School of Art, Falmouth College of Art and Design and then Falmouth College of Arts until it received
taught degree-awarding powers, and the right to use the title "University College", in March 2005.

 In April 2008, University College Falmouth merged with Dartington College of Arts, adding a range of Performance courses to its portfolio.

 On 27 November 2012, a communication was released to the staff and students and local press that 'University College Falmouth is to be granted full university
status in a move that will further its ambition to become one of the top five arts universities in the world.'

 On 9 December 2012, the University College was officially granted full university status by the Privy Council.

 Falmouth University was judged by The Sunday Times to be the UK’s top arts university from 2015-2017.
</item 4>

<item 5>
Alpha Gamma
Alpha Gamma (ΑΓ) was an American collegiate fraternity.

 It was founded at Cumberland University in Lebanon, Tennessee in 1867.

 About 21 chapters developed, with prominent chapters at Washington & Jefferson College, Trinity University, Mercersburg College, Southwestern Presbyterian
University, Cumberland University, and the West Virginia University.

 The fraternity's badge consisted of a golden shield with a globe, six stars, and a pennant with "ΑΓ."
</item 5>

<item 6>
Inklings
The Inklings was an informal literary discussion group associated with the University of Oxford, England, for nearly two decades between the early 1930s and
late 1949.

 The Inklings were literary enthusiasts who praised the value of narrative in fiction and encouraged the writing of fantasy.
</item 6>

<item 7>
We Are Poets
We Are Poets is a documentary film directed by Daniel Lucchesi and Alex Ramseyer-Bache.

 It follows six young poets from Leeds Young Authors performance poetry group in Leeds on their visit to the international Poetry Slam Competition in Washington
D.C, United States.

 It premièred at Europe's largest Documentary Film festival - Doc/Fest in Sheffield's Showroom Cinema on 11 June 2011, where it won the coveted Sheffield Youth
Jury Award.
</item 7>

<item 8>
Adam Fox
Adam Fox (1883 – 1977), Canon, was the Dean of Divinity at Magdalen College, Oxford.

 He was one of the first members of the "Inklings", a literary group which also included C.S. Lewis and J.R.R. Tolkien.

 Between 1938 and 1942 he was Professor of Poetry.

 Later he became Canon of Westminster Abbey and he is buried there in Poet's Corner.

 He was also Warden (Headmaster) of the famous Radley College.
</item 8>

<item 9>
Eagle Street College
The Eagle Street College was an informal literary society established in 1885 at the home of James William Wallace in Eagle Street, Bolton, to read and discuss
literary works, particularly the poetry of Walt Whitman, (1819–91).

 The group subsequently became known as the Bolton Whitman Fellowship or Whitmanites.

 Its founder members were Wallace, Dr John Johnston and Fred Wild.

 The group held an annual 'Whitman Day' celebration around the 31 May which was the poet's birthday.
</item 9>

<item 10>
Litbits
Litbits is a literary discussion radio programme and podcast, hosted by academic Adam Smyth and literary journalist James Kidd.

 Guests on the programme have included Polly Russell, Paul Myerscough, Alex von Tunzelmann, Samantha Ellis, Jonathan Beckman, Hallie Rubenhold, Steve Rose, Joe
Brooker, Dennis Duncan, Dinah Roe, Giles Milton and Liane Strauss.

 Subjects covered include Rooms and Procrastination.
</item 10>
=== END ITEM id=d8d97df2d88ebd76 ===

=== ITEM id=7637bb6329dda672 ===
Gold sentences and facts:
Sentence 0: California State Polytechnic University, Pomona (CPP, Cal Poly Pomona, or Cal Poly) is a public polytechnic university located in Pomona, California, United States.
  Fact 0.0: California State Polytechnic University, Pomona is also called CPP, Cal Poly Pomona, or Cal Poly.
  Fact 0.1: California State Polytechnic University, Pomona is a public polytechnic university.
  Fact 0.2: California State Polytechnic University, Pomona is located in Pomona, California, United States.
Sentence 1:  It is one of two polytechnics in the California State University system.
  Fact 1.0: California State Polytechnic University, Pomona is one of two polytechnics in the California State University system.
Sentence 2:  Ельцина, "Uralʹskiĭ federalʹnyĭ universitet imeni pervogo Prezidenta Rossii B.N. Yelʹtsina", often shortened to UrFU, УрФУ ) (formed by a merger of the Ural State Technical University and Ural State University) is one of the leading educational institutions in the Ural region.
  Fact 2.0: The university is named after the first President of Russia, B.N. Yeltsin.
  Fact 2.1: The full name of the university is 'Uralʹskiĭ federalʹnyĭ universitet imeni pervogo Prezidenta Rossii B.N. Yelʹtsina'.
  Fact 2.2: The university, Ural Federal University, is often shortened to UrFU or УрФУ.
  Fact 2.3: Ural Federal University was formed by a merger of the Ural State Technical University and Ural State University.
  Fact 2.4: Ural Federal University is one of the leading educational institutions in the Ural region.
Sentence 3:  Ural Federal University acts as a research and innovation center of the Ural region and has close cooperation with the Russian Academy of Sciences.
  Fact 3.0: Ural Federal University acts as a research and innovation center of the Ural region.
  Fact 3.1: Ural Federal University has close cooperation with the Russian Academy of Sciences.

Retrieved items:
<item 1>
Cal Poly Pomona Broncos women's basketball
The Cal Poly Pomona Broncos women's basketball team is the women's basketball team that represents California State Polytechnic University, Pomona, in Pomona,
California.

 The school's team currently competes in the California Collegiate Athletic Association.
</item 1>

<item 2>
Julian A. McPhee
Julian Aeneas McPhee (February 7, 1896 – November 10, 1967) was the sixth university president of California Polytechnic State University, San Luis Obispo (Cal
Poly SLO) from 1933 to 1966 and the first president of California State Polytechnic University, Pomona (Cal Poly Pomona) from 1938 to 1966.
</item 2>

<item 3>
1981 Cal Poly Pomona Broncos football team
The 1981 Cal Poly Pomona Broncos football team represented California State Polytechnic University, Pomona during the 1981 NCAA Division II football season.

 Cal Poly Pomona competed in the California Collegiate Athletic Association (CCAA).
</item 3>

<item 4>
Cal Poly Pomona University Library
The Cal Poly Pomona University Library is the main library on the campus of California State Polytechnic University, Pomona (Cal Poly Pomona).

 It has in its collection 670,580 books; 6,883 serial subscriptions and 10,417 audiovisual materials.
</item 4>

<item 5>
Ural Federal University
The Ural Federal University named after the first President of Russia B. N. Yeltsin (Russian: Уральский федеральный университет имени первого Президента России
Б.Н.

 Ельцина, "Uralʹskiĭ federalʹnyĭ universitet imeni pervogo Prezidenta Rossii B.N. Yelʹtsina", often shortened to UrFU, УрФУ ) (formed by a merger of the Ural
State Technical University and Ural State University) is one of the leading educational institutions in the Ural region.

 Ural Federal University acts as a research and innovation center of the Ural region and has close cooperation with the Russian Academy of Sciences.

 Training of students is carried out in four main areas of knowledge and 108 academic majors.

 The links between the university and intermediate educational institutions are actively developed.
</item 5>

<item 6>
California State Polytechnic University, Pomona
California State Polytechnic University, Pomona (CPP, Cal Poly Pomona, or Cal Poly) is a public polytechnic university located in Pomona, California, United
States.

 It is one of two polytechnics in the California State University system.
</item 6>

<item 7>
Cal Poly Pomona College of Environmental Design
The Cal Poly Pomona College of Environmental Design (CENV) is a college part of the California State Polytechnic University, Pomona (Cal Poly Pomona).

 The college houses over 1,600 students; making it one of largest environmental design programs in the United States.

 The college offers bachelor's degrees in five departments, as well as three master's degree programs.

 It is the only academic unit within the California State University system to be associated with a Pritzker Prize laureate (often referred to as "The Nobel
Prize in Architecture").
</item 7>

<item 8>
Cal Poly Pomona College of Engineering
The Cal Poly Pomona College of Engineering is the engineering college at California State Polytechnic University, Pomona (Cal Poly Pomona or Cal Poly) located
in Pomona, California, United States.

 Well known for its "learn by doing" philosophy, the college's motto is: ""making imagination real"".

 Cal Poly has one of the top engineering college among public schools in California and, with over 5,600 students (as of fall 2015), it is also the largest
engineering college in Southern California, the second largest college of engineering in the California State University system, and the seventeenth largest
engineering college in the United States.

 In the 2017 "U.S. News & World Report" the College of Engineering is ranked the 5th best undergraduate program among public universities (11th overall) in the
West for Master's-granting universities, and "has one of the top ranked engineering programs, and graduates roughly 1 of every 14 engineers in the state of
California."
</item 8>

<item 9>
International Polytechnic High School
International Polytechnic High School, commonly abbreviated merely as iPoly, is a public college preparatory demonstration high school (9-12) located on the
California State Polytechnic University, Pomona (Cal Poly Pomona) campus and operated by the Los Angeles County Office of Education (LACOE) in conjunction with
the College of Education and Integrative Studies at the university.

 iPoly's curriculum is fully accredited by the Western Association of Schools and Colleges and is approved by the University of California and California State
University.

 It maintains a unique affiliation with the Los Angeles County High School for the Arts (LACHSA), which is also run by LACOE.

 Since iPoly does not lie in a fixed school district, it draws students from throughout Los Angeles, San Bernardino, Orange and Riverside counties.

 The majority of students come from the Pomona and San Gabriel valleys.

 In 2009 and 2013, iPoly was honored as a California Distinguished School by the California Department of Education.
</item 9>

<item 10>
Cal Poly Pomona Broncos
The Cal Poly Pomona Broncos or Cal Poly Broncos are the athletic sports teams for the California State Polytechnic University, Pomona (Cal Poly Pomona).

 Cal Poly Pomona has 10 varsity sports teams and offers student participation in a wide range of sports including soccer, volleyball, track and field,
basketball, softball, and baseball.

 Cal Poly Pomona participates at the NCAA's Division II (DII) level in the California Collegiate Athletic Association (CCAA).

 The Broncos have 65 CCAA championships and 14 NCAA National Championships.

 Current and former Cal Poly athletes have won 7 Olympic medals (3 gold, 1 silver, and 3 bronze).
</item 10>
=== END ITEM id=7637bb6329dda672 ===

=== ITEM id=d273980b75ceae64 ===
Gold sentences and facts:
Sentence 0: Corn Ranch is a spaceport in the West Texas town of Van Horn, Texas, where flight tests of the New Shepard are carried out by Blue Origin.
  Fact 0.0: Corn Ranch is a spaceport.
  Fact 0.1: Corn Ranch is located in the West Texas town of Van Horn, Texas.
  Fact 0.2: Flight tests of the New Shepard are carried out at Corn Ranch.
  Fact 0.3: Flight tests of the New Shepard are carried out by Blue Origin.
Sentence 1: Blue Origin is an American privately funded aerospace manufacturer and spaceflight services company set up by Amazon.com founder Jeff Bezos with its headquarters in Kent, Washington.
  Fact 1.0: Blue Origin is an American privately funded aerospace manufacturer.
  Fact 1.1: Blue Origin is a spaceflight services company.
  Fact 1.2: Blue Origin was set up by Amazon.com founder Jeff Bezos.
  Fact 1.3: Blue Origin has its headquarters in Kent, Washington.

Retrieved items:
<item 1>
Blue Origin
Blue Origin is an American privately funded aerospace manufacturer and spaceflight services company set up by Amazon.com founder Jeff Bezos with its
headquarters in Kent, Washington.

 The company is developing technologies to enable private human access to space with the goal to dramatically lower costs and increase reliability.

 Blue Origin is employing an incremental approach from suborbital to orbital flight, with each developmental step building on its prior work.

 The company motto is ""Gradatim Ferociter"", Latin for "Step by Step, Ferociously".

 Blue Origin is developing a variety of technologies, with a focus on rocket-powered Vertical Takeoff and Vertical Landing (VTVL) vehicles for access to
suborbital and orbital space.

 The company's name refers to the blue planet, Earth, as the point of origin.
</item 1>

<item 2>
Corn Ranch
Corn Ranch is a spaceport in the West Texas town of Van Horn, Texas, where flight tests of the New Shepard are carried out by Blue Origin.

 The 165,000-acre (670 km²) land parcel was purchased by Internet billionaire Jeff Bezos.

 The first flight test took place on November 13, 2006 with the goal of providing commercial tourist flights.
</item 2>

<item 3>
Rick Husband Amarillo International Airport
Rick Husband Amarillo International Airport (IATA: AMA, ICAO: KAMA, FAA LID: AMA) is a public airport six miles (10 km) east of downtown Amarillo, in Potter and
Randall Counties, Texas, United States.

 The airport was renamed in 2003 after NASA astronaut and Amarillo native Rick Husband, who died in the Space Shuttle "Columbia" disaster in February of that
year.
</item 3>

<item 4>
H-T Ranch
H-T Ranch is a historic ranch complex 10 mi west of Amidon, North Dakota.

 The ranch originally consisted of ten buildings; however, only the ranch house and log barn survive.

 The ranch served as the headquarters of the Little Missouri Cattle Company, which was run by Arthur Clark Hidekoper.

 Hidekober established the ranch in the 1880s, and by the end of the decade, it had become the "most notable" ranch in the area.

 The ranch house, called Shackford, was built in 1896; its uncommon style resembles a bungalow but also borrows from other styles such as the Stick style.

 By 1906, the "Fargo Forum" described the ranch as "the biggest and most important [ranch] in the state" and "one of the largest horse raising outlets in the
world".

 In the same year, Hidekoper sold the 70,000 acre ranch; the sale was the largest land deal in North Dakota history.

 After the sale, a land company reduced the ranch to 5000 acre ; it was later used as a dude ranch in the 1920s.
</item 4>

<item 5>
Texas Air &amp; Space Museum
The Texas Air & Space Museum is an aviation museum located near Rick Husband Amarillo International Airport in Amarillo, Texas.

 The museum displays civilian and military aircraft as well as a wide range of air and space artifacts.
</item 5>

<item 6>
M7 Aerospace
M7 Aerospace LP is an aerospace company with its headquarters on the property of San Antonio International Airport in Uptown San Antonio, Texas, United States.
</item 6>

<item 7>
SyberJet Aircraft
SyberJet Aircraft is the manufacturer of the SJ30 light business jet.

 The company's headquarters is in Cedar City, Utah adjacent to the Cedar City Regional Airport with additional engineering offices and manufacturing, service,
repair and fatigue test facilities near and on the San Antonio International Airport in San Antonio, Texas.
</item 7>

<item 8>
GoJet Airlines
GoJet Airlines LLC is a company headquartered in Bridgeton, Missouri, United States.

 Wholly owned by Trans States Holdings, it has 1670 employees.

 It operates commuter feeder services under the United Express and Delta Connection names.

 Go Jet Airlines has crew bases at Lambert-Saint Louis International Airport, O'Hare International Airport, Denver International Airport, and Raleigh-Durham
International Airport.

 Flights are currently operated out of United's hubs at O'Hare International Airport and Denver International Airport, as well as Lambert-Saint Louis
International Airport.

 GoJet's Delta Connection flights currently operate out of Lambert-Saint Louis International Airport , Detroit Metropolitan Airport and Raleigh-Durham
International Airport.

 Its call sign "Lindbergh" is named after aviation pioneer Charles Lindbergh, who flew the "Spirit of St. Louis" solo across the Atlantic Ocean in 1927, the
first person to do so.
</item 8>

<item 9>
Grand Sky
Grand Sky is an UAS (unmanned aerial systems)-specific business and aviation park located at Grand Forks Air Force Base near Grand Forks, North Dakota.

 A groundbreaking for the development was held in September 2015, making it the first UAS business and aviation park in the U.S.

 The park consists of 217 acres and includes access to the Grand Forks Air Force Base runway for large UAS flights.
</item 9>

<item 10>
Northern California TRACON
Northern California TRACON (NCT) (Terminal Radar Approach Control), or NorCal TRACON for short, is an air traffic control facility that provides safety alerts,
separation, and sequencing of air traffic arriving, departing, and transiting the airspace and airports in Northern California.

 Located in Rancho Cordova near Sacramento, NCT controls airspace over 19000 square miles, and serves Reno International Airport, Sacramento International
Airport, San Jose International Airport, Oakland International Airport, and San Francisco International Airport, plus 19 other smaller airports with air traffic
control towers.

 NCT is the 3rd busiest TRACON in America.

 NorCal TRACON is the step between local control (in an airport's control tower) and Air Route Traffic Control Center (ARTCC), in this case, Oakland Center
(ICAO code: ZOA).

 San Francisco International Airport is the 2nd largest airport in California and the largest airport serving Northern California.
</item 10>
=== END ITEM id=d273980b75ceae64 ===

=== ITEM id=403c213f13ffd184 ===
Gold sentences and facts:
Sentence 0: Aspidistra is a genus of flowering plants in the family Asparagaceae, subfamily Nolinoideae, native to eastern and southeastern Asia, particularly China and Vietnam.
  Fact 0.0: Aspidistra is a genus of flowering plants.
  Fact 0.1: Aspidistra is in the family Asparagaceae.
  Fact 0.2: Aspidistra is in the subfamily Nolinoideae.
  Fact 0.3: Aspidistra is native to eastern Asia.
  Fact 0.4: Aspidistra is native to southeastern Asia.
  Fact 0.5: Aspidistra is particularly native to China.
  Fact 0.6: Aspidistra is particularly native to Vietnam.
Sentence 1: Cyrtanthus is a genus of perennial, herbaceous and bulbous plants in the family Amaryllidaceae, subfamily Amaryllidoideae.
  Fact 1.0: Cyrtanthus is a genus of perennial plants.
  Fact 1.1: Cyrtanthus is a genus of herbaceous plants.
  Fact 1.2: Cyrtanthus is a genus of bulbous plants.
  Fact 1.3: Cyrtanthus is in the family Amaryllidaceae.
  Fact 1.4: Cyrtanthus is in the subfamily Amaryllidoideae.

Retrieved items:
<item 1>
Ammocharis
Ammocharis is a small genus from sub-Saharan Africa, in the Amaryllidaceae family (subfamily Amaryllidoideae) which includes 7 species distributed in Africa.

 The plant grows as a succulent, above-ground bulb, preferring seasonally wet, hot, sandy soils and full sun.
</item 1>

<item 2>
Traubiinae
Traubiinae is a subtribe of plants classified under the tribe Hippeastreae.

 It belongs to the subfamily Amaryllidoideae of the Amaryllis family (Amaryllidaceae).
</item 2>

<item 3>
Nerine masoniorum
Nerine masoniorum is a species of flowering plant in the family Amaryllidaceae, subfamily Amaryllidoideae, native to the eastern Cape Province of South Africa.

 It is a bulbous perennial.

 "N. masoniorum" belongs to the group of nerines that have narrow evergreen foliage.

 The thread-like leaves reach a length of 25 cm or more.

 The flowering stem is 15–25 cm tall, with up to 11 flowers arranged in an umbel.

 Each flower has six narrow pink tepals with wavy edges.

 It flowers in late summer in cultivation, the first of the nerines to do so.
</item 3>

<item 4>
Clinantheae
Clinantheae is a tribe (in the family Amaryllidaceae, subfamily Amaryllidoideae), where it forms part of the Andean clade, one of two American clades.

 The tribe was described in 2000 by Alan Meerow "et al." as a result of a molecular phylogenetic study of the American Amaryllidoideae.

 This demonstrated that the Stenomesseae tribe, including the type genus "Stenomesson" was polyphyletic.

 Part of the tribe segregated with the Eucharideae and were submerged into it, while the other part formed a unique subclade.

 Since the type species of "Stenomesson" was not part of the second sublclade it was necessary to form a new name for the remaining species together with the
other genera that remained.

 This was "Clinanthus", the oldest name for these species, and consequently the tribe Clinantheae.
</item 4>

<item 5>
Hippeastrinae
Hippeastrinae is a subtribe of plants classified under the tribe Hippeastreae.

 It belongs to the subfamily Amaryllidoideae of the Amaryllis family (Amaryllidaceae).
</item 5>

<item 6>
Hymenocallideae
Hymenocallideae is a tribe (in the family Amaryllidaceae, subfamily Amaryllidoideae), where it forms part of the Andean clade, one of two American clades.

 The tribe was originally recognised by both Meerow (1995) and the Muller-Doblies' (1996).

 Its phylogenetic position within the Amaryllidoideae was established by Meerow "et al." in 2000, while in-depth infratribal relationships were established in
2002.
</item 6>

<item 7>
Cyrtanthus
Cyrtanthus is a genus of perennial, herbaceous and bulbous plants in the family Amaryllidaceae, subfamily Amaryllidoideae.
</item 7>

<item 8>
Lycoris radiata
Lycoris radiata, known as red spider lily, red magic lily, or equinox flower, is a plant in the amaryllis family, Amaryllidaceae, subfamily Amaryllidoideae.

 Originally from China, Korea and Nepal, it was introduced into Japan and from there to the United States and elsewhere.

 It is considered naturalized in Seychelles and in the Ryukyu Islands.

 It flowers in the late summer or autumn, often in response to heavy rainfall.

 The common name hurricane lily refers to this characteristic, as do other common names, such as resurrection lily; these may be used for the genus as a whole.
</item 8>

<item 9>
Aspidistra
Aspidistra is a genus of flowering plants in the family Asparagaceae, subfamily Nolinoideae, native to eastern and southeastern Asia, particularly China and
Vietnam.

 They grow in shade under trees and shrubs.

 Their leaves arise more or less directly from ground level, where their flowers also appear.

 The number of species known has increased considerably from the 1980s onwards, with around 100 accepted as of July 2013 . "

Aspidistra elatior" is common worldwide as a foliage house plant that is very tolerant of neglect.

 It and other species can also be grown in shade outside, where they are generally hardy to -5 C .
</item 9>

<item 10>
Amarylloidinae
Amarylloidinae is a now obsolete informal name for an "infrafamily" within the Amaryllidaceae (Amaryllis) family, erected by Hamilton Traub.

 This grouping was designed to fill a perceived gap between the formal rank of subfamily and tribes.

 In his treatment of this family, he divided it first into four subfamiles.

 Within subfamily Amarylloideae he then divided his sixteen tribes into two infrafamilies, Amarylloidinae (12 tribes) and Pancratioidinae (4 tribes), both of
which were subsequently demonstrated to be polyphyletic, and hence were abandoned by Dahlgren, who used no rank between family and tribe.

 On the other hand, he also used a much more restricted Amaryllidaceae corresponding to Traub's subfamily Amarylloideae.

 Thus Traub's Amarylloideae most closely resembles subfamily Amaryllidoideae "sensu" APGIII.
</item 10>
=== END ITEM id=403c213f13ffd184 ===

=== ITEM id=7f1edc95971f75aa ===
Gold sentences and facts:
Sentence 0:  During his early career, Cage starred in a variety of films such as "Valley Girl" (1983), "Racing with the Moon" (1984), "Birdy" (1984), "Peggy Sue Got Married" (1986), "Raising Arizona" (1987), "Moonstruck" (1987), "Vampire's Kiss" (1989), "Wild at Heart" (1990), "Fire Birds" (1990), "Honeymoon in Vegas" (1992), and "Red Rock West" (1993).
  Fact 0.0: During his early career, Cage starred in a variety of films.
  Fact 0.1: Cage starred in Valley Girl, released in 1983.
  Fact 0.2: Cage starred in Racing with the Moon, released in 1984.
  Fact 0.3: Cage starred in Birdy, released in 1984.
  Fact 0.4: Cage starred in Peggy Sue Got Married, released in 1986.
  Fact 0.5: Cage starred in Raising Arizona, released in 1987.
  Fact 0.6: Cage starred in Moonstruck, released in 1987.
  Fact 0.7: Cage starred in Vampire's Kiss, released in 1989.
  Fact 0.8: Cage starred in Wild at Heart, released in 1990.
  Fact 0.9: Cage starred in Fire Birds, released in 1990.
  Fact 0.10: Cage starred in Honeymoon in Vegas, released in 1992.
  Fact 0.11: Cage starred in Red Rock West, released in 1993.
Sentence 1: Moonstruck is a 1987 American romantic comedy film directed by Norman Jewison and written by John Patrick Shanley.
  Fact 1.0: Moonstruck is a 1987 American film.
  Fact 1.1: Moonstruck is a romantic comedy film.
  Fact 1.2: Moonstruck was directed by Norman Jewison.
  Fact 1.3: Moonstruck was written by John Patrick Shanley.
Sentence 2:  It is about a widowed 37-year-old Italian-American woman (Cher) who falls in love with her fiancé's (Danny Aiello) estranged, hot-tempered younger brother (Nicolas Cage).
  Fact 2.0: Moonstruck is about a widowed 37-year-old Italian-American woman played by Cher.
  Fact 2.1: The woman falls in love with her fiancé's estranged, hot-tempered younger brother.
  Fact 2.2: The fiancé is played by Danny Aiello.
  Fact 2.3: The younger brother is played by Nicolas Cage.

Retrieved items:
<item 1>
Cool and the Crazy
Cool and the Crazy is a 1994 American drama film written and directed by Ralph Bakshi and starring Jared Leto and Alicia Silverstone.

 The story revolves around an unhappily married couple in the late 1950s who both lead separate affairs.

 The film was Bakshi's first feature-length live-action film, being primarily known as a director of animated films which heavily utilize live-action sequences,
such as "Fritz the Cat", "Heavy Traffic", "Wizards", "American Pop" and "The Lord of the Rings".
</item 1>

<item 2>
In Country
In Country is a 1989 American drama film produced and directed by Norman Jewison, starring Bruce Willis and Emily Lloyd.

 The screenplay by Frank Pierson and Cynthia Cidre was based on the novel by Bobbie Ann Mason.

 The original music score was composed by James Horner.

 Willis earned a best supporting actor Golden Globe nomination for his role.
</item 2>

<item 3>
Nicolas Cage
Nicolas Kim Coppola (born January 7, 1964), known professionally as Nicolas Cage, is an American actor, director and producer.

 During his early career, Cage starred in a variety of films such as "Valley Girl" (1983), "Racing with the Moon" (1984), "Birdy" (1984), "Peggy Sue Got
Married" (1986), "Raising Arizona" (1987), "Moonstruck" (1987), "Vampire's Kiss" (1989), "Wild at Heart" (1990), "Fire Birds" (1990), "Honeymoon in Vegas"
(1992), and "Red Rock West" (1993).
</item 3>

<item 4>
Moonstruck
Moonstruck is a 1987 American romantic comedy film directed by Norman Jewison and written by John Patrick Shanley.

 It is about a widowed 37-year-old Italian-American woman (Cher) who falls in love with her fiancé's (Danny Aiello) estranged, hot-tempered younger brother
(Nicolas Cage).

 Vincent Gardenia and Olympia Dukakis play supporting roles.
</item 4>

<item 5>
Chloe (film)
Chloe is a 2009 erotic thriller film directed by Atom Egoyan, a remake of the 2003 French film "Nathalie...".

 It stars Julianne Moore, Liam Neeson, and Amanda Seyfried in the title role.

 Its screenplay was written by Erin Cressida Wilson, based on the earlier French film, written by Anne Fontaine.
</item 5>

<item 6>
The Night Before (1988 film)
The Night Before is a 1988 film starring Keanu Reeves and Lori Loughlin.

 Reeves plays Winston Connelly, the so-called high school nerd and vice president of the astronomy club.

 Loughlin plays Tara Mitchell, the pretty and popular head cheerleader who also happens to be the local police chief's daughter.

 The Tagline was: "You lost your father's car, sold your prom date and a guy called 'Tito' wants you dead.

 It's a date that's the time of your life."

 It was filmed and set entirely in Los Angeles, California.
</item 6>

<item 7>
Dying of the Light (film)
Dying of the Light is a 2014 American psychological thriller film written and directed by Paul Schrader and starring Nicolas Cage, Anton Yelchin and Irène Jacob
about a government agent who must track down and kill a terrorist before he loses his full memory from a disease.

 It was released theatrically and through VOD formats by Lionsgate on December 5, 2014.

 The film received extremely negative reviews, with controversy surrounding the heavy tampering and reediting of the footage by the studio, who denied Schrader
final-cut privilege and led him and principal members of the cast to disown the released version and campaign against it.
</item 7>

<item 8>
Where Do You Go (Cher song)
"Where Do You Go" is a song written by Sonny Bono.

 It was released as the first single by Cher in the later quarter of 1965 for her second album "The Sonny Side of Cher".

 It fell short of the "Billboard" Hot 100's top 20, but still earned Cher a moderate success, by reaching the Top 40.

 It was followed by the U.S. #2 smash hit "Bang Bang (My Baby Shot Me Down)" which saw a release early the following year.

 It was a bigger hit in Canada, where it reached #5 on the singles chart.
</item 8>

<item 9>
Deadfall (1993 film)
Deadfall is a 1993 crime drama film directed by Christopher Coppola.

 Coppola co-wrote the script with Nick Vallelonga.

 The film stars Michael Biehn, Nicolas Cage, Charlie Sheen, James Coburn, and Peter Fonda.

 It is also the prime influence on the song 'Deadfall', written by the American hardcore punk band Snot.

 A prequel/sequel, "Arsenal", starring Nicolas Cage as his character Eddie King, was released in 2017.
</item 9>

<item 10>
Cher filmography
Throughout her acting career, Cher has mainly in comedy, drama, and romance films.

 She has appeared in thirteen films, including two as a cameo.

 She has also appeared in one starring theater role, numerous television commercials and directed a piece of the motion picture "If These Walls Could Talk" in
1996 and some of her music videos of the Geffen-era in late 1980s and in early 1990s.

 Cher has starred in various international television commercials, as well as high-profile print advertising for Lori Davis (1992).

 Before she started her film career, she had a couple of hits in the 1960s, as a solo artist, and with her ex-husband Sonny Bono as the couple Sonny & Cher.
</item 10>
=== END ITEM id=7f1edc95971f75aa ===

=== ITEM id=958bf165c9382153 ===
Gold sentences and facts:
Sentence 0:  Obertraun is a popular holiday destination offering activities such as skiing, snowboarding in the winter and mountain biking, swimming and kayaking in the summer.
  Fact 0.0: Obertraun is a popular holiday destination.
  Fact 0.1: Obertraun offers skiing and snowboarding in the winter.
  Fact 0.2: Obertraun offers mountain biking in the summer.
  Fact 0.3: Obertraun offers swimming and kayaking in the summer.
Sentence 1:  The name "Salzkammergut" literally means "Estate of the Salt Chamber" and derives from the Imperial Salt Chamber, the authority charged with running the precious salt mines of the Habsburg Monarchy.
  Fact 1.0: The name 'Salzkammergut' literally means 'Estate of the Salt Chamber'.
  Fact 1.1: The name 'Salzkammergut' derives from the Imperial Salt Chamber.
  Fact 1.2: The Imperial Salt Chamber was the authority charged with running the precious salt mines of the Habsburg Monarchy.

Retrieved items:
<item 1>
Appenzeller cheese
Appenzeller cheese is a hard cow's-milk cheese produced in the Appenzell region of northeast Switzerland.

 A herbal brine, sometimes incorporating wine or cider, is applied to the wheels of cheese while they cure, which flavors and preserves the cheese while
promoting the formation of a rind.
</item 1>

<item 2>
Salzkammergut
The Salzkammergut is a resort area located in Austria.

 It stretches from the City of Salzburg eastwards along the Austrian Alpine Foreland and the Northern Limestone Alps to the peaks of the Dachstein Mountains,
spanning the federal states of Upper Austria, Salzburg, and Styria.

 The main river of the region is the Traun, a right tributary of the Danube.

 The name "Salzkammergut" literally means "Estate of the Salt Chamber" and derives from the Imperial Salt Chamber, the authority charged with running the
precious salt mines of the Habsburg Monarchy.

 It is a UNESCO World Heritage Site.
</item 2>

<item 3>
Aspen/Snowmass
Aspen Snowmass is a winter resort complex located in Pitkin County in western Colorado in the United States.

 Owned and operated by the Aspen Skiing Company it comprises four skiing/snowboarding areas on four adjacent mountains in the vicinity of the towns of Aspen and
Snowmass Village.

 The four areas collectively form one of the most famous winter resorts in the world and are annually the destination for visitors from all over the world.
</item 3>

<item 4>
Upper Saxon German
Upper Saxon (German: "Obersächsisch" ) is an East Central German dialect spoken in much of the modern German State of Saxony and in the adjacent parts of
Saxony-Anhalt and Thuringia.

 Though colloquially called "Saxon" ("Sächsisch" ), it is not to be confused with the Low Saxon dialect group in Northern Germany.

 Upper Saxon is closely linked to the Thuringian dialect spoken in the adjacent areas to the west.
</item 4>

<item 5>
Obertraun
Obertraun, Upper Austria is a village in the Salzkammergut, a region in Austria.

 It is located near the Hallstätter See (Hallstatt Lake) and Hoher Dachstein.

 Obertraun is a popular holiday destination offering activities such as skiing, snowboarding in the winter and mountain biking, swimming and kayaking in the
summer.
</item 5>

<item 6>
List of Austrian mountain climbers
This is a list of Austrian mountaineers.

 On 5 of the 14 Eight-thousanders Austrians have made the first ascent, more than any other nation can claim.

 Out of the total of 9 Austrian mountaineers who made first ascents of Eight-thousanders, 3 have been members of the Edelweiss Club Salzburg, an association of
mountaineers founded in Salzburg in 1881
</item 6>

<item 7>
South Bergish
South Bergish (German: "Südbergische Dialekte ") or Upper Bergish (German: "Oberbergische Dialekte ") is a group of German dialects of the Bergisches Land
Region East of the Rhine and approximately south of the Wupper and north of the Sieg.

 These dialects are part of the Ripuarian group and thus are also called East Ripuarian.

 Ripuarian dialects are also spoken west of the Rhine up to the German border, and in some small areas next to the respective borders in Belgium and in the
Netherlands.

 Ripuarian Bergish dialects belong to the Middle German group, and thus are varieties of High German, where they belong to the northmost ones.

 In the North, they border to the East Bergish and the West Bergish aka Low Bergish language groups, which are part of the Low Franconian group like Dutch.
</item 7>

<item 8>
Hundsheimer Berg
Hundsheimer Berg is a hill located in the Hundsheimer Berge hill range in Lower Austria, Austria close to the border with Slovakia.

 Its peak is 481 m above sea level which makes it the highest hill in the Hundsheimer Berge.

 The north slopes of the Hundsheimer Berg are covered with forests while the southern slopes have a steppe vegetation.

 Since 1965, a nature reserve with the same name, "Hundsheimer Berg", is located on these southern slopes.
</item 8>

<item 9>
Soviet Unterzoegersdorf
Soviet Unterzoegersdorf (German: "Sowjet-Unterzögersdorf" ) is a fictitious country created by the art/technology/theory group monochrom.

 It is the "last existing appanage republic of the USSR", located inside the Republic of Austria.

 Unterzögersdorf is partially based on an existing village well-known to Johannes Grenzfurthner, the creator of the concept.

 As a kid and teenager, Grenzfurthner spent a lot of time at his grandparents' farm in the small village of Unterzögersdorf (a cadastral municipality of
Stockerau).

 His grandparents' stories about the Naziism, WWII and the Soviet occupation in allied-occupied Austria (1945-1955) form the inspirational basis of the
long-term project "Soviet Unterzoegersdorf".
</item 9>

<item 10>
Appenzell Alps
The Appenzell Alps (German: "Appenzeller Alpen" ) are a mountain range in Switzerland on the northern edge of the Alps.

 They extend into the cantons of Appenzell Ausserrhoden, Appenzell Innerrhoden and St. Gallen and are bordered by the Glarus Alps to the west and the Rätikon to
the south-east.
</item 10>
=== END ITEM id=958bf165c9382153 ===

=== ITEM id=50e8e939dcf47f20 ===
Gold sentences and facts:
Sentence 0:  There is no optimal strategy involved as play entirely rests upon a random drawing of the cards.
  Fact 0.0: There is no optimal strategy involved in the game.
  Fact 0.1: Play in the game entirely rests upon a random drawing of the cards.
Sentence 1: Diplomacy is a strategic board game created by Allan B. Calhamer in 1954 and released commercially in 1959.
  Fact 1.0: Diplomacy is a strategic board game.
  Fact 1.1: Diplomacy was created by Allan B. Calhamer.
  Fact 1.2: Diplomacy was created in 1954.
  Fact 1.3: Diplomacy was released commercially in 1959.
Sentence 2:  Its main distinctions from most board wargames are its negotiation phases (players spend much of their time forming and betraying alliances with other players and forming beneficial strategies) and the absence of dice and other game elements that produce random effects.
  Fact 2.0: Diplomacy's main distinctions from most board wargames include its negotiation phases.
  Fact 2.1: In Diplomacy's negotiation phases, players spend much of their time forming and betraying alliances with other players.
  Fact 2.2: In Diplomacy's negotiation phases, players spend much of their time forming beneficial strategies.
  Fact 2.3: Diplomacy's main distinctions from most board wargames include the absence of dice and other game elements that produce random effects.

Retrieved items:
<item 1>
Dungeons &amp; Dragons Basic Game
The Dungeons & Dragons Basic Game is an introductory version of "Dungeons & Dragons" ("D&D") role-playing game packaged in the form of a board game.

 The original game was released in 2004 by Wizards of the Coast and was designed by Jonathan Tweet, one of the "D&D" 3rd edition designers.

 A new version of this game was released in September 2006.
</item 1>

<item 2>
Simultaneous action selection
Simultaneous action selection, or SAS, is a game mechanic that occurs when players of a game take action (such as moving their pieces) at the same time.

 An example of a game that uses this type of movement is the game Diplomacy.

 Typically, a "secret yet binding" method of committing to one's move is necessary, so that as players' moves are revealed and implemented, others do not change
their moves in light of the new information.

 Thus, in Diplomacy, players write down their moves and then reveal them simultaneously.

 Because no player gets the first move, this potentially arbitrary source of advantage is not present.

 It is also possible for simultaneous movement games to proceed relatively quickly, because players are acting at the same time, rather than waiting for their
turn.

 Simultaneous action selection is easily implemented in card games such as Apples to Apples in which players simply select cards and throw them face-down into
the center.
</item 2>

<item 3>
Kolejka (game)
Kolejka ( ; Polish for "queue" or "line (of people)") is a board game designed by Karol Madaj and launched in February 2011 by Poland's Institute of National
Remembrance.

 It was designed to teach young generations about the hardships under communism (1945–1989), particularly the difficulty of acquiring consumer goods in the
shortage economy.

 The game has been described as inspired by "Monopoly".
</item 3>

<item 4>
Attack! (board game)
Attack!

 is a board game created by Glenn Drover and published by Eagle Games in 2003.

 It is a light war game that is midway between "Risk" and "Axis and Allies" in complexity.

 The game is loosely set in the 1930s and includes plastic pieces featuring tanks, planes, infantry, and artillery.

 "Attack!"

 won the Origins Award for "Best Historical Game 2003".
</item 4>

<item 5>
Diplomacy (game)
Diplomacy is a strategic board game created by Allan B. Calhamer in 1954 and released commercially in 1959.

 Its main distinctions from most board wargames are its negotiation phases (players spend much of their time forming and betraying alliances with other players
and forming beneficial strategies) and the absence of dice and other game elements that produce random effects.

 Set in Europe before the beginning of World War I, "Diplomacy" is played by two to seven players, each controlling the armed forces of a major European power
(or, with fewer players, multiple powers).

 Each player aims to move his or her few starting units and defeat those of others to win possession of a majority of strategic cities and provinces marked as
"supply centers" on the map; these supply centers allow players who control them to produce more units.
</item 5>

<item 6>
Uncle Wiggily
Uncle Wiggily Longears is the main character of a series of children's stories by American author Howard R. Garis.

 He began writing the stories for the "Newark News" in 1910.

 Garis penned an Uncle Wiggily story every day (except Sundays) for more than 30 years, and published 79 books within the author's lifetime.

 According to his obituary in the "Chicago Tribune", a walk in the woods in Verona, New Jersey was his inspiration.

 The books featured work by several illustrators, notably Lansing Campbell.

 Other illustrators of the series included George L. Carlson, Louis Wisa, Elmer Rache, Edward Bloomfield, Lang Campbell and Mary and Wallace Stover.
</item 6>

<item 7>
Kill Doctor Lucky
Kill Doctor Lucky is a humorous board game designed by James Ernest and released in 1996 by Cheapass Games.

 In 1998, "Kill Doctor Lucky" won the Origins Award for "Best Abstract Board Game of 1997".
</item 7>

<item 8>
Uncle Wiggily (board game)
Uncle Wiggily Game is a track board game based on a character in a series of children's books by American writer Howard Roger Garis.

 The game is of the "racing" variety in the style of the European "Goose Game".

 Players advance along the track from Uncle Wiggily's Bungalow to Dr. Possum's House.

 There is no optimal strategy involved as play entirely rests upon a random drawing of the cards.

 The game was first published by Milton Bradley in 1916 and has seen several editions with minor modifications over the years.

 "Uncle Wiggily" remains one of the first and favorite games of childhood, and, with "Candy Land", is considered a classic juvenile American board game.
</item 8>

<item 9>
Can't Stop (board game)
Can't Stop is a board game designed by Sid Sackson originally published by Parker Brothers in 1980, and was long out of print in the United States.

 It was reprinted by Face 2 Face Games in 2007.

 An iOS version was developed by Playdek and released in 2012.

 The goal of the game is to "claim" (get to the top of) three of the columns before any of the other players can.

 But the more that the player risks rolling the dice during a turn, the greater the risk of losing the advances made during that turn.
</item 9>

<item 10>
Alhambra (board game)
Alhambra (German: Der Palast von Alhambra , literally "The Palace of the Alhambra") is a 2003 tile-based German-style board game designed by Dirk Henn.

 It was originally published in Germany by Queen Games in a language-interdependent version; an English-specific version was released in North America by the
now-defunct Überplay.

 The game is an Arabian-themed update, set during the construction of the Alhambra palace in 14th century Granada, of the 1998 stock trading board game "Stimmt
So!"

, which in turn was an update of the 1992 mafia influence board game "Al Capone"; the original version was subsequently released as "".
</item 10>
=== END ITEM id=50e8e939dcf47f20 ===

=== ITEM id=a0433d6e41100fae ===
Gold sentences and facts:
Sentence 0: The Karlsschrein (English: Shrine of Charlemagne ) in Aachen Cathedral was made in Aachen at the command of Frederick II, Holy Roman Emperor and completed in 1215, after Frederick II's grandfather, Frederick Barbarossa had exhumed Charlemagne's bones from their resting place in the Palatine Chapel, Aachen in 1165.
  Fact 0.0: The Karlsschrein, English name Shrine of Charlemagne, is in Aachen Cathedral.
  Fact 0.1: The Karlsschrein was made in Aachen.
  Fact 0.2: The Karlsschrein was made at the command of Frederick II, Holy Roman Emperor.
  Fact 0.3: The Karlsschrein was completed in 1215.
  Fact 0.4: Frederick II's grandfather was Frederick Barbarossa.
  Fact 0.5: Frederick Barbarossa exhumed Charlemagne's bones from their resting place in the Palatine Chapel, Aachen.
  Fact 0.6: Frederick Barbarossa exhumed Charlemagne's bones in 1165.
  Fact 0.7: The exhumation of Charlemagne's bones happened before the Karlsschrein was completed.
Sentence 1: Frederick I (German: "Friedrich I" , Italian: "Federico I" ; 1122 – 10 June 1190), also known as Frederick Barbarossa (Italian: "Federico Barbarossa" ), was the Holy Roman Emperor from 1155 until his death.
  Fact 1.0: Frederick I's German name is "Friedrich I".
  Fact 1.1: Frederick I's Italian name is "Federico I".
  Fact 1.2: Frederick I lived from 1122 to 10 June 1190.
  Fact 1.3: Frederick I is also known as Frederick Barbarossa.
  Fact 1.4: Frederick Barbarossa's Italian name is "Federico Barbarossa".
  Fact 1.5: Frederick I was the Holy Roman Emperor from 1155 until his death.

Retrieved items:
<item 1>
Harald II of Denmark
Harald II of Denmark (died 1018) was King of Denmark from 1014 until his death in 1018.

 He was the youngest son of Sweyn Forkbeard and Gunhild of Wenden, and was regent while his father was fighting Ethelred the Unready in England.

 He inherited the Danish throne in 1014, and held it while his brother, the later king Cnut the Great conquered England.

 After his death in 1018, he was succeeded by Cnut the Great.

 Little detail is known about Harald II.
</item 1>

<item 2>
Ernest Augustus, King of Hanover
Ernest Augustus (5 June 1771 – 18 November 1851) was King of Hanover from 20 June 1837 until his death.

 He was the fifth son and eighth child of King George III of the United Kingdom and Hanover.

 As a fifth son, Ernest seemed unlikely to become a monarch, but none of his four elder brothers had a legitimate son who survived infancy.

 The Salic Law, which barred succession to or through a female, prevailed in Hanover; therefore, when his elder brother King William IV died in 1837, Ernest
succeeded him as King of Hanover.

 In the United Kingdom the succession to the monarchy was determined by primogeniture and his niece Victoria succeeded to the throne, thus ending the personal
union between the British Isles and Hanover that had existed since 1714.
</item 2>

<item 3>
List of Holy Roman Empresses
Holy Roman Empress or Empress of the Holy Roman Empire is the title given to the consort (wife) or regent of the Holy Roman Emperor.

 The elective dignity of Holy Roman Emperor was restricted to males only, therefore there was never a Holy Roman Empress regnant, though women such as Theophanu
or Maria Theresa of Austria, who controlled the power of rule, served as de facto Empresses regnant.
</item 3>

<item 4>
Gottlieb Muffat
Gottlieb Muffat (April 1690 – 9 December 1770), son of Georg Muffat, served as "Hofscholar" under Johann Fux in Vienna from 1711 and was appointed to the
position of third court organist at the "Hofkapelle" in 1717.

 He acquired additional duties over time including the instruction of members of the Imperial family, among them the future Empress Maria Theresa.

 He was promoted to second organist in 1729 and first organist upon the accession of Maria Theresa to the throne in 1741.

 He retired from official duties at the court in 1763.
</item 4>

<item 5>
Eric II of Denmark
Eric II the Memorable (Danish: "Erik II Emune" ) (  1090 – 18 September 1137) was king of Denmark between 1134 and 1137.

 Eric was an illegitimate son of Eric I of Denmark, who ruled Denmark from 1095 to 1103.

 Eric the Memorable rebelled against his uncle Niels of Denmark, and was declared king in 1134.

 He punished his adversaries severely, and rewarded his supporters handsomely.

 He was killed by a subject in 1137 and was promptly succeeded by his nephew Eric III of Denmark.
</item 5>

<item 6>
Karl Anselm, 4th Prince of Thurn and Taxis
Karl Anselm, 4th Prince of Thurn and Taxis, full German name: "Karl Anselm Fürst von Thurn und Taxis" (2 June 1733, Frankfurt am Main, Free Imperial City of
Frankfurt, Holy Roman Empire – 13 November 1805, Winzer bei Regensburg, Electorate of Bavaria, Holy Roman Empire) was the fourth Prince of Thurn and Taxis,
Postmaster General of the Imperial Reichspost, and Head of the Princely House of Thurn and Taxis from 17 March 1773 until his death on 13 November 1805.

 Karl Anselm served as "Prinzipalkommissar" at the Perpetual Imperial Diet in Regensburg for Joseph II, Holy Roman Emperor and Francis II, Holy Roman Emperor
from 1773 to 1797.
</item 6>

<item 7>
Karlsschrein
The Karlsschrein (English: Shrine of Charlemagne ) in Aachen Cathedral was made in Aachen at the command of Frederick II, Holy Roman Emperor and completed in
1215, after Frederick II's grandfather, Frederick Barbarossa had exhumed Charlemagne's bones from their resting place in the Palatine Chapel, Aachen in 1165.
</item 7>

<item 8>
Prince Christian of Hanover (born 1985)
Prince Christian of Hanover (Christian Heinrich Clemens Paul Frank Peter Welf Wilhelm-Ernst Friedrich Franz Prince of Hanover, Duke of Brunswick and Lüneburg;
born 1 June 1985) is the younger son of Ernst August, Prince of Hanover, and his first wife Chantal Hochuli.

 He is the second in the line of succession to the former Hanoverian throne, after his elder brother Prince Ernst August.

 As a descendant of George III of the United Kingdom, Christian is also in the line of succession to the British throne.
</item 8>

<item 9>
Frederick I, Holy Roman Emperor
Frederick I (German: "Friedrich I" , Italian: "Federico I" ; 1122 – 10 June 1190), also known as Frederick Barbarossa (Italian: "Federico Barbarossa" ), was the
Holy Roman Emperor from 1155 until his death.

 He was elected King of Germany at Frankfurt on 4 March 1152 and crowned in Aachen on 9 March 1152.

 He became King of Italy in 1155 and was crowned Roman Emperor by Pope Adrian IV on 18 June 1155.

 Two years later, the term "sacrum " ("holy") first appeared in a document in connection with his Empire.

 He was later formally crowned King of Burgundy, at Arles on 30 June 1178.

 He was named "Barbarossa " by the northern Italian cities which he attempted to rule: Barbarossa means "red beard" in Italian; in German, he was known as
"Kaiser Rotbart ", which has the same meaning.
</item 9>

<item 10>
Eric III of Denmark
Eric III Lamb (Danish: "Erik III Lam" ) (  1120 – 27 August 1146) was the King of Denmark from 1137 until 1146.

 He was the grandson of Eric I of Denmark and the nephew of Eric II of Denmark, whom he succeeded on the throne.

 He abdicated in 1146, as the first and only Danish monarch to do so.

 His succession led to a period of civil war between Sweyn III, Canute V, and Valdemar I.
</item 10>
=== END ITEM id=a0433d6e41100fae ===

=== ITEM id=40d052af6a8a25f9 ===
Gold sentences and facts:
Sentence 0: Lithocarpus is a genus in the beech family Fagaceae, differing from "Quercus" in the erect spikes of insect-pollinated male flowers and the short styles with punctate stigmas on the female flowers.
  Fact 0.0: Lithocarpus is a genus in the beech family Fagaceae.
  Fact 0.1: Lithocarpus differs from "Quercus" in the erect spikes of insect-pollinated male flowers.
  Fact 0.2: Lithocarpus differs from "Quercus" in the short styles with punctate stigmas on the female flowers.
Sentence 1:  It contains 17 species of shrubs and small trees that are native from southern Florida to Mexico and South America.
  Fact 1.0: Lithocarpus contains 17 species of shrubs and small trees.
  Fact 1.1: The Lithocarpus species are native from southern Florida to Mexico and South America.

Retrieved items:
<item 1>
Strobilanthes
Strobilanthes is a genus of about 350 species of flowering plants in the family Acanthaceae, mostly native to tropical Asia and Madagascar, but with a few
species extending north into temperate regions of Asia.

 Many species are cultivated for their 2-lipped, hooded flowers in shades of blue, pink, white and purple.

 Most are frost-tender and require protection in frost-prone areas.
</item 1>

<item 2>
Duranta
Duranta is a genus of flowering plants in the verbena family, Verbenaceae.

 It contains 17 species of shrubs and small trees that are native from southern Florida to Mexico and South America.

 They are commonly cultivated as hedges and ornamental plants.
</item 2>

<item 3>
Lithocarpus platycarpus
Lithocarpus platycarpus is a species of plant in the Fagaceae family.

 It is a tree endemic to Java in Indonesia.

 It is an endangered species threatened by habitat loss.

.
</item 3>

<item 4>
Lithocarpus crassinervius
Lithocarpus crassinervius is a species of plant in the Fagaceae family.

 It is a tree endemic to Java in Indonesia.

 It is an endangered species threatened by habitat loss.
</item 4>

<item 5>
Cassytha
Cassytha is a genus of some two dozen species of obligately parasitic vines in the family Lauraceae.

 Superficially, and in some aspects of their ecology, they closely resemble plants in the unrelated genus "Cuscuta", the dodders.

 When fruit and flowers are absent in the field, the physical resemblance is so close that few people without technical training can tell them apart.

 In this respect and in their ecology the two genera present a spectacular example of convergent evolution.

 In its divergence from habits typical of the Lauraceae, "Cassytha" also presents examples of mosaic evolution
</item 5>

<item 6>
Lithocarpus kostermansii
Lithocarpus kostermansii is a species of plant in the Fagaceae family.

 It is a tree endemic to Java in Indonesia.

 It is an endangered species threatened by habitat loss.
</item 6>

<item 7>
Lithocarpus glaber
Lithocarpus glaber, the Japanese oak, is a tree species in the genus "Lithocarpus" found in Japan, China and Taïwan.
</item 7>

<item 8>
Lithocarpus
Lithocarpus is a genus in the beech family Fagaceae, differing from "Quercus" in the erect spikes of insect-pollinated male flowers and the short styles with
punctate stigmas on the female flowers.

 The World Checklist (see link below) accepts 334 species, though some other texts suggest as few as 100 species.

 About 100 Asian species of the genus were formerly treated in the genus "Pasania".

 All are native to east and southeast Asia.

 These Asian species do not have a well-known English vernacular name, though the generic term stone oak has been proposed.
</item 8>

<item 9>
Graph query row:
entity: Lithocarpus glaber
relationship: LOCATED_IN
shared_connection: China
other_relationship: LOCATED_IN
related_entity: Duranta erecta
</item 9>

<item 10>
Duranta erecta
Duranta erecta is a species of flowering shrub in the verbena family Verbenaceae, native from Mexico to South America and the Caribbean.

 It is widely cultivated as an ornamental plant in tropical and subtropical gardens throughout the world, and has become naturalized in many places.

 It is considered an invasive species in Australia, China, South Africa and on several Pacific Islands.

 Common names include golden dewdrop, pigeon berry, and skyflower.

 In Mexico, the native Nahuatl name for the plant is xcambocoché.

 In Tonga it is known as mavaetangi (tears of departure).
</item 10>
=== END ITEM id=40d052af6a8a25f9 ===
