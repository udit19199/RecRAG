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

=== ITEM id=3f98e94e9fb82cf4 ===
Gold sentences and facts:
Sentence 0: Vaugier appeared in Need for Speed : Carbon as Nikki . She is the girlfriend of the antagonist in the game ( love interest of the protagonist ) .
  Fact 0.0: Vaugier appeared in Need for Speed: Carbon as Nikki.
  Fact 0.1: Nikki is the girlfriend of the antagonist in Need for Speed: Carbon.
  Fact 0.2: Nikki is the love interest of the protagonist in Need for Speed: Carbon.
Sentence 1: Year Title Role Notes 2006 Need for Speed : Carbon Nikki Voice , Mo - Cap , & Likeness 2015 Dying Light Jade Mo - Cap 2015 Hellraid Maine , The Female Kin Voice
  Fact 1.0: The filmography table lists a 2006 role as Nikki in Need for Speed: Carbon, with notes Voice, Mo-Cap, & Likeness.
  Fact 1.1: The filmography table lists a 2015 role as Jade in Dying Light, with notes Mo-Cap.
  Fact 1.2: The filmography table lists a 2015 role as Maine, The Female Kin in Hellraid, with notes Voice.

Retrieved items:
<item 1>
Emmanuelle Vaugier
She has an older brother , Jason Vaugier . Career ( edit ) Television ( edit ) Vaugier played the role of young , upstart model Maria Alcobar in a 1996 episode
of Highlander : The Series . In 2004 , she played Lindsey Kellogg on the short - lived Fox Television series North Shore . She has also had guest - star spots
on Veronica Mars , Supernatural , The Outer Limits , Higher Ground , Andromeda , and Charmed . She also appeared in the Emmy Award nominated miniseries The
Beach Boys : An American Family where she portrayed Mike Love 's wife , Suzanne . Vaugier also completed a starring role in MOW Veiled Truth for the Lifetime
Network . She is also known for her role during the first two seasons of One Tree Hill , as Nicki , Jake 's ( Bryan Greenberg ) ex-girlfriend and mother to
their child , Jenny . From 2005 - 2015 , Vaugier guest starred in Twelve episodes of CBS 's hit sitcom Two and a Half Men as Mia , a ballet teacher with whom
Charlie Harper fell in love and almost married .
</item 1>

<item 2>
The Divergent Series
Cast ( edit ) On March 11 , 2013 , it was announced that the roles of Tori , Christina , and Caleb had gone to Maggie Q , Zoë Kravitz , and Ansel Elgort ,
respectively . Ray Stevenson , Jai Courtney and Aaron Eckhart were announced to be in talks to join the cast on March 15 , 2013 , Stevenson and Courtney joined
the cast as Marcus Eaton and Eric , respectively . That same day , Miles Teller was cast as Peter . The studio 's announcement included a bit of casting news :
Mekhi Phifer will play Dauntless leader Max . Kate Winslet was announced to be in talks on January 24 , 2013 . Later it was confirmed that she would portray
Jeanine Matthews . Talking about playing the negative character for the first time , Winslet said , `` I 'm no idiot . The idea went through my head that I have
never played a baddie before , I was almost kind of surprised . '' Since Winslet joined the filming late , she used that distance from her co-stars to appear
aloof on the first day of her shoot .
</item 2>

<item 3>
Krysten Ritter
The The Last International Playboy Ozzy 2008 27 Dresses Gina the goth 2008 What Happens in Vegas Kelly 2009 Glock Beretta Short film 2009 Confessions of a
Shopaholic Suze Cleath - Stuart She 's Out of My League Patty How to Make Love to a Woman Lauren 2011 Killing Bono Gloria 2011 Life Happens Kim Also writer and
co-producer 2011 Margaret Shopgirl 2012 BuzzKill Nicole 2012 Vamps Stacy Daimen 2012 Refuge Amy 2014 Listen Up Philip Melanie 2014 Veronica Mars Gia Goodman
2014 Asthma Ruby 2014 Search Party Christy 2014 Big Eyes DeAnn 2017 Hero , The The Hero Lucy Television ( edit ) Year Title Role Notes Whoopi Brynn Episode : ``
The Squatters '' One Life to Live Kay 4 episodes Law & Order Tracy Warren Episode : `` Everybody Loves Raimondo 's '' Tanner on Tanner Saleswoman 2 episodes
2005 Jonny Zero Quinn Episode : `` Pilot '' 2005 -- 2006 Veronica Mars Gia Goodman Recurring role ; 8 episodes 2006 Bedford Diaries ,
</item 3>

<item 4>
During a Deadpool 2 stunt, Joi “SJ” Harris, portraying Domino, was not wearing a helmet because the character had none in the scene and there was no time to
make one that fit under her wig. She had two days of rehearsal and five attempts on the accident day. An experienced stunt double was available, but Harris was
chosen partly because her skin tone matched actress Zazie Beetz’s—a decision stunt professionals criticized, noting motorcycle-racing experience did not ensure
stunt expertise. Filming
</item 4>

<item 5>
Emmanuelle Vaugier
Stine 's The Haunting Hour Abigail Episode 1.06 : `` The Red Dress '' 2011 Killer Mountain Kate TV movie 2012 It 's Christmas , Carol ! Carol Huffman TV movie
2013 The Mentalist Melissa Enfield Episode 5.16 : `` There Will Be Blood '' 2013 Clara 's Deadly Secret Helen TV movie 2013 Hidden Away Alexandra Cole /
Stephanie Bennett TV movie 2015 My Stepdaughter Jill TV movie 2015 Mistresses Niko Recurring role ; 6 episodes 2016 Love In Paradise Heather Twain TV movie 2016
Stranger in the House Jade TV movie 2016 His Double Life Linda TV movie 2016 Love in the Vineyard ( Heart Felt ) Molly TV movie 2017 Rogue Regan Recurring role
; 7 episodes 2017 Washed Away Gabrielle Video Games ( edit ) Year Title Role Notes 2006 Need for Speed : Carbon Nikki Voice , Mo - Cap , & Likeness 2015 Dying
Light Jade Mo - Cap 2015 Hellraid Maine , The Female Kin Voice References ( edit ) Jump up ^ `` Archived copy '' .
</item 5>

<item 6>
8 Mile (film)
Mekhi Phifer as Future , Jimmy 's closest friend and the rap battle host Brittany Murphy as Alex Latourno , Jimmy 's love interest Kim Basinger as Stephanie ,
Jimmy and Lily 's mother Anthony Mackie as Papa Doc Michael Shannon as Greg Buehl , Stephanie 's boyfriend Eugene Byrd as Wink Evan Jones as Cheddar Bob , Jimmy
's friend Omar Benson Miller as Sol George , Jimmy 's friend and Iz 's brother De'Angelo Wilson as DJ Iz , Jimmy 's friend and Sol 's brother Taryn Manning as
Janeane , Jimmy 's ex-girlfriend Proof as Lil ' Tic Xzibit as Mike ( Male Lunch Truck Rapper ) Craig Chandler as Paul Obie Trice as Male parking lot rapper
Chloe Greenfield as Lily , Jimmy 's sister John Singleton as Bouncer Miz Korona as Vanessa ( Female Lunch Truck Rapper ) Brandon T. Jackson ( uncredited ) as a
Chin Tiki club - foe Music ( edit ) Main article : Music from and Inspired by the Motion Picture 8 Mile Music from and Inspired by the Motion Picture 8 Mile is
the soundtrack to 8 Mile .
</item 6>

<item 7>
The Divergent Series
`` I wanted to break it and say , ' It 's OK , I 'm really fun . I promise . ' But I thought , just for today , I 'd let them think that I am a complete bitch .
'' On March 25 , 2013 , Ben Lamb was cast as Edward , Ben Lloyd - Hughes as Will , and Christian Madsen as Al . In April 2013 , Ashley Judd and Tony Goldwyn
joined the cast as Natalie and Andrew Prior , Beatrice 's parents . In 2014 , the cast for the second film was announced . On May 12 , 2014 , it was announced
that Octavia Spencer joined the cast as Amity representative Johanna Reyes . Late May 2014 , Suki Waterhouse and Jonny Weston were cast as Marlene and Edgar
respectively . Early June 2014 , Stephanie Leigh Schlund announced that she was cast in the film as a member of the Amity , and Naomi Watts and Daniel Dae Kim
joined the cast as Evelyn Johnson and Jack Kang . On June 9 , 2014 , Rosa Salazar joined the cast as Lynn .
</item 7>

<item 8>
Casting announcements included Maggie Q as Tori, Zoë Kravitz as Christina, Ansel Elgort as Caleb, Ray Stevenson as Marcus Eaton, Jai Courtney as Eric, Miles
Teller as Peter, and Mekhi Phifer as Dauntless leader Max. Kate Winslet joined as Jeanine Matthews after talks began in January 2013. She noted she had never
played a villain before and used her late arrival on set to seem aloof around her co-stars.
</item 8>

<item 9>
The Eboladrome track was modified after an unexploded WWII bomb was found. Series one’s American test driver, former NASCAR racer Mike Skinner, was replaced by
British driver Abbie Eaton after poor reception. Skinner set a 1:26.2 lap in a BMW M2; a Porsche 918 later beat it by two seconds. In hypercar tests, Jérôme
d’Ambrosio timed laps, and Hammond’s 918 beat Clarkson’s McLaren P1, prompting a
</item 9>

<item 10>
Emmanuelle Vaugier
She appeared in the film 40 Days and 40 Nights , which starred Josh Hartnett . She had starring roles in Suddenly Naked , in which she plays a Latin pop
sensation ; the TV - movie Mindstorm , a science - fiction thriller ; and Ripper , a psychological thriller about students who mysteriously disappear after
enrolling in a class about serial killers . She stars also in the sequel to Mirrors , which is directed by Victor Garcia . Maxim magazine featured her on the
cover of the February 2006 issue in the United States . Three months later , she landed spot # 31 on their annual Hot 100 list . Games ( edit ) Vaugier appeared
in Need for Speed : Carbon as Nikki . She is the girlfriend of the antagonist in the game ( love interest of the protagonist ) .
</item 10>
=== END ITEM id=3f98e94e9fb82cf4 ===

=== ITEM id=e842182ae371201f ===
Gold sentences and facts:
Sentence 0: From September 1947 , Toyota 's small - sized vehicles were sold under the name `` Toyopet '' ( トヨペット ) . The first vehicle sold under this name was the Toyopet SA , but it also included vehicles such as the Toyopet SB light truck , Toyopet Stout light truck , Toyopet Crown , Toyopet Master , and the Toyopet Corona . The word `` Toyopet ( Japanese article ) '' was a nickname given to the Toyota SA due to its small size , as the result of a naming contest the Toyota Company organized in 1947 . However , when Toyota eventually entered the American market in 1957 with the Crown , the name was not well received due to connotations of toys and pets . The name was soon dropped for the American market , but continued in other markets until the mid-1960s .
  Fact 0.0: From September 1947, Toyota's small-sized vehicles were sold under the name 'Toyopet' (トヨペット).
  Fact 0.1: The first vehicle sold under the name Toyopet was the Toyopet SA.
  Fact 0.2: Toyopet vehicles also included the Toyopet SB light truck.
  Fact 0.3: Toyopet vehicles also included the Toyopet Stout light truck.
  Fact 0.4: Toyopet vehicles also included the Toyopet Crown.
  Fact 0.5: Toyopet vehicles also included the Toyopet Master.
  Fact 0.6: Toyopet vehicles also included the Toyopet Corona.
  Fact 0.7: The word 'Toyopet' was a nickname given to the Toyota SA due to its small size.
  Fact 0.8: The Toyopet nickname resulted from a naming contest the Toyota Company organized in 1947.
  Fact 0.9: When Toyota entered the American market in 1957 with the Crown, the Toyopet name was not well received due to connotations of toys and pets.
  Fact 0.10: The Toyopet name was soon dropped for the American market.
  Fact 0.11: The Toyopet name continued in other markets until the mid-1960s.

Retrieved items:
<item 1>
Toyota
Japan : Toyota Motor Corporation . January 29 , 2013 . Retrieved February 3 , 2014 . Jump up ^ `` December 2013 Sales Chart '' ( Press release ) . USA : Toyota
. January 2013 . Retrieved January 10 , 2014 . Jump up ^ `` TMC Announces Results for December 2014 and CY2014 '' . Jump up ^ `` Toyota U.S. Sales Summary '' (
Press release ) . USA : Toyota . Retrieved February 14 , 2015 . Jump up ^ `` TMC Announces Results for December 2015 and CY2015 '' . Jump up ^ `` TMC Announces
Results for December 2016 and CY2016 '' . Jump up ^ Toyota : A history of the First 50 Years . Toyota Motor Corporation . 1988 . ISBN 0 - 517 - 61777 - 3 . ,
p461 . ^ Jump up to : Harden , Blaine ( July 13 , 2008 ) . `` Japan 's Killer Work Ethic '' . Washington Post Foreign Service .
</item 1>

<item 2>
Toyota
Toyota Technical Center , Ann Arbor Charter Township , Michigan Toyota has a large presence in the United States with six major assembly plants in Huntsville ,
Alabama , Georgetown , Kentucky , Princeton , Indiana , San Antonio , Texas , Buffalo , West Virginia , and Blue Springs , Mississippi . Toyota had a joint -
venture operation with General Motors at New United Motor Manufacturing Inc. in Fremont , California , which began in 1984 and ended in 2009 . It still has a
joint venture with Subaru at Subaru of Indiana Automotive , Inc. in Lafayette , Indiana , which started in 2006 . In these assembly plants , the Camry and the
Tundra are manufactured , among others . Toyota marketing , sales , and distribution in the US are conducted through a separate subsidiary , Toyota Motor Sales
, U.S.A. , Inc . It has started producing larger trucks , such as the new Tundra , to go after the large truck market in the United States . Toyota is also
pushing hybrid vehicles in the US such as the Prius , Camry Hybrid , Highlander Hybrid , and various Lexus products . Currently , Toyota has no plans to offer
diesel motor options in its North American products , including the light - duty pickup trucks .
</item 2>

<item 3>
Toyota
American car manufacturers had considered small economy cars to be an entry - level product , and their small vehicles employed a low level of quality to keep
the price low . Conservative Toyota held on to rear - wheel - drive designs for longer than most ; while a clear first in overall production they were only
third in production of front - wheel - drive cars in 1983 , behind Nissan and Honda . In part due to this , Nissan 's Sunny managed to squeeze by the Corolla in
numbers built that year . In 1982 , the Toyota Motor Company and Toyota Motor Sales merged into one company , the Toyota Motor Corporation . Two years later ,
Toyota entered into a joint venture with General Motors called the New United Motor Manufacturing , Inc , NUMMI , operating an automobile - manufacturing plant
in Fremont , California . The factory was an old General Motors plant that had been closed for two years . Toyota then started to establish new brands at the
end of the 1980s , with the launch of their luxury division Lexus in 1989 .
</item 3>

<item 4>
Toyota
North America ( edit ) Main article : Toyota Motor Engineering & Manufacturing North America Toyota Motor North America headquarters is located in Plano , Texas
, and operates as a holding company in North America . Its manufacturing headquarters is located in Erlanger , Kentucky , and is known as Toyota Motor
Engineering & Manufacturing North America . Toyota Canada Inc. has been in production in Canada since 1983 with an aluminium wheel plant in Delta , British
Columbia , which currently employs a workforce of roughly 260 . Its first vehicle assembly plant , in Cambridge , Ontario , since 1988 , now produces Corolla
compact cars , Matrix crossover vehicles , and Lexus RX 350 luxury SUVs , with a workforce of 4,300 workers . Its second assembly operation in Woodstock ,
Ontario , began manufacturing the RAV4 late in 2008 . In 2006 , Toyota 's subsidiary Hino Motors opened a heavy duty truck plant , also in Woodstock , employing
45 people and producing 2000 trucks annually .
</item 4>

<item 5>
Toyota
However , when Toyota eventually entered the American market in 1957 with the Crown , the name was not well received due to connotations of toys and pets . The
name was soon dropped for the American market , but continued in other markets until the mid-1960s . 1960s -- 1970s ( edit ) By the early 1960s , the US had
begun placing stiff import tariffs on certain vehicles . The so - called `` chicken tax '' of 1964 placed a 25 % tax on imported light trucks . In response to
the tariff , Toyota , Nissan Motor Co. and Honda Motor Co. began building plants in the US by the early 1980s . 1980s ( edit ) With over 40 million sold , the
Corolla is one of the most popular and best selling cars in the world . Toyota received its first Japanese Quality Control Award at the start of the 1980s and
began participating in a wide variety of motorsports . Due to the 1973 oil crisis , consumers in the lucrative US market began turning to small cars with better
fuel economy .
</item 5>

<item 6>
Toyota
Toyota - wikipedia Toyota Jump to : navigation , search For other uses , see Toyota ( disambiguation ) . Toyota Motor Corporation hkl headquarters in Toyota
City , Japan Type Public ( K.K. ) Traded as TYO : 7203 LSE : TYT NYSE : TM TOPIX Core 30 Component Industry Automotive Founded August 28 , 1937 ; 80 years ago (
1937 - 08 - 28 ) Headquarters Toyota , Aichi , Japan Area served Worldwide Key people Kiichiro Toyoda ( Chairman ) Shigeru Hayakawa ( Vice Chairman ) Takeshi
Uchiyamada ( President ) Products Automobiles , luxury vehicles , commercial vehicles , engines Production output 16,985,870 units ( CY 2015 ) Services Banking
, financing , leasing Revenue ¥ 28.40 trillion ( 2016 ) Operating income ¥ 2.85 trillion ( 2016 ) Profit ¥ 2.31 trillion ( 2016 ) Total assets ¥ 47.42 trillion
( 2016 ) Total equity ¥ 17.22 trillion ( 2016 ) Owner Toyota Industries ( 6.73 % ) Nippon Life ( 3.
</item 6>

<item 7>
Toyota
1990s ( edit ) In the 1990s , Toyota began to branch out from producing mostly compact cars by adding many larger and more luxurious vehicles to its lineup ,
including a full - sized pickup , the T100 ( and later the Tundra ) , several lines of SUVs , a sport version of the Camry , known as the Camry Solara , and the
Scion brand . They would also launch newer and arguably more iconic iterations of their sports cars , namely the MR2 , Celica , and Supra during this era , all
of which have already become icons of the 1990s . With a major presence in Europe , due to the success of Toyota Team Europe , the corporation decided to set up
Toyota Motor Europe Marketing and Engineering , TMME , to help market vehicles in the continent . Two years later , Toyota set up a base in the United Kingdom ,
TMUK , as the company 's cars had become very popular among British drivers . Bases in Indiana , Virginia , and Tianjin were also set up . In 1999 , the company
decided to list itself on the New York and London Stock Exchanges .
</item 7>

<item 8>
Toyota
Akio Toyoda became the new president and CEO of the company on June 23 , 2009 , by replacing Katsuaki Watanabe , who became the new vice chairman by replacing
Katsuhiro Nakagawa . Worldwide presence ( edit ) The Toyota Camry is assembled in several facilities around the world including Australia , China , Taiwan ,
Japan , Malaysia , the Philippines , Pakistan , Russia , Thailand , India , Vietnam , and the United States Toyota has factories in most parts of the world ,
manufacturing or assembling vehicles for local markets in Japan , Australia , India , Sri Lanka , Canada , Indonesia , Poland , South Africa , Turkey , Colombia
, the United Kingdom , the United States , France , Brazil , Portugal , and more recently , Argentina , Czech Republic , Mexico , Malaysia , Thailand , Pakistan
, Egypt , China , Vietnam , Venezuela , the Philippines , and Russia .
</item 8>

<item 9>
Toyota
The principle of jidoka , which means the machine stops itself when a problem occurs , became later a part of the Toyota Production System . Looms were built on
a small production line . In 1929 , the patent for the automatic loom was sold to the British company Platt Brothers , generating the starting capital for the
automobile development . Mass production of Toyoda automated loom , displayed at the Toyota Museum in Nagakute - cho , Aichi - gun , Aichi Pref . Japan The
production of Toyota automobiles was started in 1933 as a division of Toyoda Automatic Loom Works devoted to the production of automobiles under the direction
of the founder 's son , Kiichiro Toyoda . Its first vehicles were the A1 passenger car and the G1 in 1935 . The Toyota Motor Co. was established as an
independent company in 1937 Toyoda Standard Sedan AA 1936 Vehicles were originally sold under the name `` Toyoda '' ( トヨダ ) , from the family name of the
company 's founder , Kiichirō Toyoda . In April 1936 , Toyoda 's first passenger car , the Model AA , was completed .
</item 9>

<item 10>
Toyota
Inside Investor . Retrieved July 24 , 2013 . Jump up ^ `` The Story of Sakichi Toyoda '' . Toyota Industries Corporation . Retrieved September 22 , 2017 . Jump
up ^ `` History Of Toyota '' . Toyota . Retrieved August 15 , 2011 . Jump up ^ `` Toyota Company History from 1867 to 1939 '' . Toyota . Archived from the
original on May 10 , 2013 . Retrieved September 11 , 2010 . Jump up ^ `` Information from a sign at the Toyota Museum in Nagakute - cho , Aichi - gun , Aichi
Pref '' . Toyota . Archived from the original on November 20 , 2012 . Retrieved December 7 , 2012 . Jump up ^ Schmitt , Bertel ( July 5 , 2011 ) . `` At Home
With The Toyodas -- And A Mysterious Cherry Tree '' . Thetruthaboutcars . Retrieved August 15 , 2011 . Jump up ^ Davis , Pedr ( 1999 ) .
</item 10>
=== END ITEM id=e842182ae371201f ===

=== ITEM id=af815319d280978a ===
Gold sentences and facts:
Sentence 0: The first mention of the expression `` Apostles ' Creed '' occurs in a letter of AD 390 from a synod in Milan and may have been associated with the belief , widely accepted in the 4th century , that , under the inspiration of the Holy Spirit , each of the Twelve Apostles contributed an article of a creed .
  Fact 0.0: The first mention of the expression Apostles' Creed occurs in a letter of AD 390.
  Fact 0.1: The letter was from a synod in Milan.
  Fact 0.2: The expression Apostles' Creed may have been associated with a belief.
  Fact 0.3: The belief was widely accepted in the 4th century.
  Fact 0.4: The belief was that, under the inspiration of the Holy Spirit, each of the Twelve Apostles contributed an article of a creed.
Sentence 1: The title Symbolum Apostolicum ( Symbol or Creed of the Apostles ) appears for the first time in a letter , probably written by Ambrose , from a Council in Milan to Pope Siricius in about AD 390 `` Let them give credit to the Creed of the Apostles , which the Roman Church has always kept and preserved undefiled '' . But what existed at that time was not what is now known as the Apostles ' Creed but a shorter statement of belief that , for instance , did not include the phrase `` maker of heaven and earth '' , a phrase that may have been inserted only in the 7th century .
  Fact 1.0: The title Symbolum Apostolicum (Symbol or Creed of the Apostles) appears for the first time in a letter.
  Fact 1.1: The letter was probably written by Ambrose.
  Fact 1.2: The letter was from a Council in Milan to Pope Siricius.
  Fact 1.3: The letter dates to about AD 390.
  Fact 1.4: The letter says: Let them give credit to the Creed of the Apostles, which the Roman Church has always kept and preserved undefiled.
  Fact 1.5: What existed at that time was not what is now known as the Apostles' Creed.
  Fact 1.6: What existed at that time was a shorter statement of belief.
  Fact 1.7: That shorter statement did not include the phrase maker of heaven and earth, for instance.
  Fact 1.8: The phrase maker of heaven and earth may have been inserted only in the 7th century.

Retrieved items:
<item 1>
Apostles' Creed
Apostles ' Creed - wikipedia Apostles ' Creed Jump to : navigation , search The rubric above this 13th - century illuminated manuscript translates `` twelve
articles of faith set out by twelve apostles '' . The Apostles ' Creed ( Latin : Symbolum Apostolorum or Symbolum Apostolicum ) , sometimes entitled Symbol of
the Apostles , is an early statement of Christian belief -- a creed or `` symbol '' . It is widely used by a number of Christian denominations for both
liturgical and catechetical purposes , most visibly by liturgical Churches of Western tradition , including the Catholic Church , Lutheranism and Anglicanism .
It is also used by Presbyterians , Methodists and Congregationalists . The Apostles ' Creed is Trinitarian in structure with sections affirming belief in God
the Father , Jesus Christ His Son and the Holy Spirit . The Apostles ' Creed was based on Christian theological understanding of the Canonical gospels , the
letters of the New Testament and to a lesser extent the Old Testament . Its basis appears to be the old Roman Creed known also as the Old Roman Symbol .
</item 1>

<item 2>
Apostles' Creed
It has been argued nonetheless that it dates from the second half of the 5th century , though no earlier . Some have suggested that the Apostles ' Creed was
spliced together with phrases from the New Testament . For instance , the phrase `` descendit ad inferos '' ( `` he descended into hell '' ) echoes Ephesians 4
: 9 , `` κατέβη εἰς τὰ κατώτερα μέρη τῆς γῆς '' ( `` he descended into the lower , earthly regions '' ) . It is of interest that this phrase first appeared in
one of the two versions of Rufinus in AD 390 and then did not appear again in any version of the creed until AD 650 . This phrase and that on the communion of
saints are articles found in the Apostles ' Creed , but not in the Old Roman Symbol nor in the Nicene Creed . Musical settings ( edit ) Musical settings of the
Symbolum Apostolorum as a motet are rare .
</item 2>

<item 3>
Apostles' Creed
Archived from the original on June 5 , 2011 . Retrieved May 19 , 2011 . ^ Jump up to : `` Apostles ' Creed '' , Dictionary of the Christian Church , Oxford
University Press , 2005 , p. 90 , ISBN 978 - 0 - 19 - 280290 - 3 . Jump up ^ Day , Gardiner Mumford ( 1963 ) , The Apostles ' Creed : an interpretation for
today , Scribner , p. 33 . Jump up ^ McGiffert , Arthur Cushman ( 2008 ) , The Apostles ' Creed : Its Origin , Its Purpose , and Its Historical Interpretation ,
p. 42 , ISBN 0 - 559 - 85199 - 5 . Jump up ^ Bettenson , Henry , ed. ( 1963 ) , Documents of the Christian Church ( 2nd ed . ) , London , p. 23 . Jump up ^
Lynch , Joseph ( 1992 ) , The Medieval Church , London and NY : Longman , p. 7 .
</item 3>

<item 4>
Apostles' Creed
180 ) . While the individual statements of belief that are included in the Apostles ' Creed -- even those not found in the Old Roman Symbol -- are found in
various writings by Irenaeus , Tertullian , Novatian , Marcellus , Rufinus , Ambrose , Augustine , Nicetas , and Eusebius Gallus , the earliest appearance of
what we know as the Apostles ' Creed was in the De singulis libris canonicis scarapsus ( `` Excerpt from Individual Canonical Books '' ) of St. Pirminius (
Migne , Patrologia Latina 89 , 1029 ff . ) , written between 710 and 714 . Bettenson and Maunder state that it is first from Dicta Abbatis Pirminii de singulis
libris canonicis scarapsus ( idem quod excarpsus , excerpt ) , c. 750 . This longer Creed seems to have arisen in what is now France and Spain . Charlemagne
imposed it throughout his dominions , and it was finally accepted in Rome , where the Old Roman Symbol or similar formulas had survived for centuries .
</item 4>

<item 5>
Apostles' Creed
1 Evangelical Lutheran Worship 5.4. 2 Lutheran Service Book 5.4. 3 Church of Denmark 5.5 United Methodist Church 6 Liturgical use in Western Christianity 6.1
Catholic Church 6.1. 1 Rite of Baptism 6.1. 2 Profession of faith at Mass 6.1. 3 Use in the Divine Office 6.2 Anglican Communion 6.2. 1 Episcopal Church (
United States ) 6.3 Lutheran Church 6.4 Methodism 7 See also 8 Notes 9 References 10 Further reading 11 External links 11.1 English translations Origin ( edit )
The title Symbolum Apostolicum ( Symbol or Creed of the Apostles ) appears for the first time in a letter , probably written by Ambrose , from a Council in
Milan to Pope Siricius in about AD 390 `` Let them give credit to the Creed of the Apostles , which the Roman Church has always kept and preserved undefiled ''
.
</item 5>

<item 6>
Apostles' Creed
Its basis appears to be the old Roman Creed known also as the Old Roman Symbol . Because of the early origin of its original form , it does not address some
Christological issues defined in the Nicene and other Christian Creeds . It thus says nothing explicitly about the divinity of either Jesus or the Holy Spirit .
This makes it acceptable to many Arians and Unitarians . Nor does it address many other theological questions which became objects of dispute centuries later .
The first mention of the expression `` Apostles ' Creed '' occurs in a letter of AD 390 from a synod in Milan and may have been associated with the belief ,
widely accepted in the 4th century , that , under the inspiration of the Holy Spirit , each of the Twelve Apostles contributed an article of a creed . Contents
( hide ) 1 Origin 2 Musical settings 3 Text in Latin 4 Text in Greek 5 English translations 5.1 Ecumenical version 5.2 Roman Catholic Church 5.2. 1 Articles 1
-- 12 5.2. 2 Articles 9 -- 10 5.3 Church of England 5.4 Lutheran Church 5.4.
</item 6>

<item 7>
Apostles' Creed
Ἀμήν . English translations ( edit ) Ecumenical version ( edit ) 15th - century Flemish tapestry illustrating the first four articles of the Creed The English
Language Liturgical Consultation ( ELLC ) is an international ecumenical group whose primary purpose is to provide ecumenically accepted texts for those who use
English in their liturgy . In 1988 it produced a translation of the Apostles ' Creed , distinguished among other things by its avoidance of the word `` his ''
in relation to God . The text is as follows : I believe in God , the Father almighty , creator of heaven and earth . I believe in Jesus Christ , God 's only Son
, our Lord , who was conceived by the Holy Spirit , born of the Virgin Mary , suffered under Pontius Pilate , was crucified , died , and was buried ; he
descended to the dead . On the third day he rose again ; he ascended into heaven , he is seated at the right hand of the Father , and he will come to judge the
living and the dead .
</item 7>

<item 8>
Apostles' Creed
But what existed at that time was not what is now known as the Apostles ' Creed but a shorter statement of belief that , for instance , did not include the
phrase `` maker of heaven and earth '' , a phrase that may have been inserted only in the 7th century . This illumination from a 13th - century manuscript shows
the apostles writing the Creed , receiving inspiration from the Holy Spirit . The account of the origin of this creed , the forerunner and principal source of
the Apostles ' Creed , as having been jointly created by the Apostles under the inspiration of the Holy Spirit , with each of the twelve contributing one of the
articles , was already current at that time . The earlier text evolved from simpler texts based on Matthew 28 : 19 , part of the Great Commission , and it has
been argued that it was already in written form by the late 2nd century ( c. 180 ) .
</item 8>

<item 9>
Apostles' Creed
200 . Jump up ^ Roman Missal Apostles ' Creed , Wentworthville : Our Lady of Mount Carmel , 2011 , retrieved September 30 , 2016 . Jump up ^ `` Rite of Baptism
'' , Lutheran Service Book . Jump up ^ `` Baptismal Covenant I '' , The United Methodist Hymnal , p. 35 . Further reading ( edit ) Cannata , Raymond F. ;
Reitano , Joshua D. ( 2013 ) , Rooted : The Apostles ' Creed , Oro Valley , AZ : Doulos Resources , ISBN 978 - 1 - 937063 - 92 - 4 Lochman , Jan Milič ( 1999 )
, `` Apostles ' Creed '' , in Fahlbusch , Erwin , Encyclopedia of Christianity , 1 , Grand Rapids : Wm . B. Eerdmans , pp. 109 -- 10 , ISBN 0 - 8028 - 2413 - 7
External links ( edit ) Wikimedia Commons has media related to Apostles ' Creed .
</item 9>

<item 10>
Apostles' Creed
7 . Jump up ^ Creeds of Christendom , with a History and Critical notes , II . The History of Creeds , Christian Classics Ethereal Library , July 13 , 2005 ,
retrieved May 19 , 2011 Jump up ^ Kelly , JND ( 1972 ) , Early Christian Creeds ( third ed . ) , London : Longman , Green & Co , pp. 398 -- 434 . Jump up ^
Bettenson , Henry ; Maunder , Chris ( 1999 ) , Documents of the Christian Church ( 3 ed . ) , New York : Oxford University Press , p. 26 . Jump up ^ `` Origin
of the Creed '' , Catholic Encyclopedia , New advent . Jump up ^ Trillhaas , Wolfgang , `` Creeds , Lutheran Attitude Toward '' , in Bodensieck , Julius , The
Encyclopedia of the Lutheran Church , A -- E , Minneapolis : Augsburg , p. 629 .
</item 10>
=== END ITEM id=af815319d280978a ===

=== ITEM id=d32c30ef2c19a288 ===
Gold sentences and facts:
Sentence 0: Windows Defender , officially called Windows Defender Antivirus in Windows 10 Creators Update , is an anti-malware component of Microsoft Windows.It was first released as a free antispyware program download for Windows XP , shipped with Windows Vista and Windows 7 and made into a full antivirus program replacing Microsoft Security Essentials as part of Windows 8 and later versions .
  Fact 0.0: Windows Defender is an anti-malware component of Microsoft Windows.
  Fact 0.1: Windows Defender is officially called Windows Defender Antivirus in Windows 10 Creators Update.
  Fact 0.2: Windows Defender was first released as a free antispyware program download for Windows XP.
  Fact 0.3: Windows Defender shipped with Windows Vista.
  Fact 0.4: Windows Defender shipped with Windows 7.
  Fact 0.5: Windows Defender was made into a full antivirus program as part of Windows 8 and later versions.
  Fact 0.6: Windows Defender replaced Microsoft Security Essentials as part of Windows 8 and later versions.

Retrieved items:
<item 1>
Windows Defender
Windows Defender - wikipedia Windows Defender Jump to : navigation , search Not to be confused with Windows Defender Security Center . This article needs
additional citations for verification . Please help improve this article by adding citations to reliable sources . Unsourced material may be challenged and
removed . ( July 2009 ) ( Learn how and when to remove this template message ) Windows Defender A component of Microsoft Windows Windows Defender Antivirus , as
appears in the Windows 10 Creators Update Details Type Antivirus software on Windows 8 and later ; formerly spyware removal software Included with Windows Vista
and later , Windows Server 2016 Also available for Windows XP and Windows Server 2003 Support status Same as the version of Windows with which it is integrated
Related components Microsoft Security Essentials Windows Defender Security Center Windows Defender , officially called Windows Defender Antivirus in Windows 10
Creators Update , is an anti-malware component of Microsoft Windows.It was first released as a free antispyware program download for Windows XP , shipped with
Windows Vista and Windows 7 and made into a full antivirus program replacing Microsoft Security Essentials as part of Windows 8 and later versions .
</item 1>

<item 2>
The text lists Windows/DOS commands for navigation, file and disk management, processes, registry, environment, contents, scripting, networking, maintenance,
boot, development, and help. Windows Defender is Microsoft’s built-in security software: initially a free antispyware product based on GIANT AntiSpyware, it
became generally available in 2006, gained antivirus protection in Windows 8, and is enabled by default unless third-party antivirus is installed. Features
include real-time and download scanning, periodic scans
</item 2>

<item 3>
Windows Defender
The first and last categories carry a link to learn more about the particular item , and the second category invites users to submit the program to Microsoft
SpyNet for analysis by community members . The Software Explorer feature has been removed from Windows Defender in Windows 7 . Notification of startup programs
that run as an administrator Windows Defender in Windows Vista automatically blocks all startup items that require administrator privileges to run ( this is
considered suspicious behavior for a startup item ) . This automatic blocking is related to the User Account Control functionality in Windows Vista , and
requires users to manually run each of these startup items each time they log in if they desire the item to run at startup . User interface In Windows Vista ,
it is possible to close the window and have the program run in the system tray while a scan is running . However , in Windows 7 , this functionality was removed
and the window must remain open while a scan is running . Windows Defender offline ( edit ) Windows Defender Offline ( formerly known as Standalone System
Sweeper Beta ) is a bootable standalone antimalware program that runs from a bootable disk and is designed to scan infected systems while their operating
systems are offline .
</item 3>

<item 4>
Windows Defender
Initially , its `` Settings '' dialog box was replaced by dedicated page in the Settings app . In Windows 10 Creators Update , Windows Defender is renamed
Windows Defender Antivirus to distinguish it from Windows Defender Security Center . The latter has become the default avenue to interface with Windows Defender
. While there is no shortcut on the Start menu for Windows Defender 's native client , it can still run . Advanced features ( edit ) Windows Defender
successfully blocks the EICAR test file Real - time protection In the Windows Defender options , the user can configure real - time protection options . Browser
integration Integration with Internet Explorer and Microsoft Edge enables files to be scanned as they are downloaded to detect malicious software inadvertently
downloaded . Although it does not integrate with non-Microsoft web browsers , Windows Defender scans for malicious downloaded files as part of its real - time
protection . Windows 10 's Anniversary Update introduced Limited Periodic Scanning , which optionally allows Windows Defender to scan a system periodically if
another antivirus app is installed . It also introduced Block at First Sight , which uses machine learning to predict whether a file is malicious . Windows
Vista - specific functionality ( edit ) Windows Defender had additional functionality in Windows Vista which was removed in subsequent versions of Windows .
</item 4>

<item 5>
Windows Defender
General availability ( edit ) On October 24 , 2006 , Microsoft released Windows Defender . It supports Windows XP and Windows Server 2003 ; however , unlike the
betas , it does not run on Windows 2000 . Conversion to antivirus ( edit ) Windows Defender was released with Windows Vista and Windows 7 , serving as their
built - in antispyware component . In Windows Vista and Windows 7 , Windows Defender was superseded by Microsoft Security Essentials , an antivirus product from
Microsoft which provided protection against a wider range of malware . Upon installation , Microsoft Security Essentials disabled and replaced Windows Defender
. In Windows 8 , Microsoft upgraded Windows Defender into an antivirus program very similar to Microsoft Security Essentials for Windows 7 and using the same
virus definition updates . MSE itself does not run on Windows versions beyond 7 . In Windows 8 and Windows 10 , Windows Defender is on by default . It switches
itself off upon installation of a third party anti-virus package . Starting with Windows 10 , Microsoft began to transfer the control of Windows Defender out of
its native client . Initially , its `` Settings '' dialog box was replaced by dedicated page in the Settings app .
</item 5>

<item 6>
Windows Defender
Windows Defender ( Beta 2 ) was released on February 13 , 2006 . It featured the program 's new name and a significant user interface redesign . The core engine
was rewritten in C++ , unlike the original GIANT - developed one , which was written in Visual Basic . This improved the application 's performance . Also ,
since beta 2 , the program works as a Windows service , unlike earlier releases , which enables the application to protect the computer even when a user is not
logged on . Beta 2 also requires Windows Genuine Advantage validation . However , Windows Defender ( Beta 2 ) did not contain some of the tools found in
Microsoft AntiSpyware ( Beta 1 ) ; Microsoft removed the System Inoculation , Secure Shredder and System Explorer tools found in MSAS ( Beta 1 ) as well as the
Tracks Eraser tool , which allowed users to easily delete many different types of temporary files related to Internet Explorer 6 , including HTTP cookies , web
cache , and Windows Media Player playback history . Microsoft later released German and Japanese versions of Windows Defender ( Beta 2 ) . General availability
( edit ) On October 24 , 2006 , Microsoft released Windows Defender .
</item 6>

<item 7>
Windows Defender
Contents ( hide ) 1 Basic features 2 History 2.1 Beta 2.2 General availability 2.3 Conversion to antivirus 3 Advanced features 4 Windows Vista - specific
functionality 5 Windows Defender Offline 6 Mitigated security vulnerability 7 See also 8 References 9 External links Basic features ( edit ) Before Windows 8 ,
Windows Defender protected against spyware . It included a number of real - time security agents that monitored several common areas of Windows for changes
which might have been caused by spyware . It also included the ability to easily remove installed ActiveX software. Windows Defender featured integrated support
for Microsoft SpyNet that allows users to report to Microsoft what they consider to be spyware , and what applications and device drivers they allow to be
installed on their system . Protection against viruses was added in Windows 8 ; Windows Defender in Windows 8 resembles Microsoft Security Essentials ( MSE )
and uses the same virus definitions . In Windows 10 , Windows Defender settings are controlled by the Settings app , and the Settings button opens the Settings
app . In the Windows 10 Anniversary Update , toast notifications appear to announce the results of a scan , even if no viruses are found .
</item 7>

<item 8>
Windows Defender
wikipedia.org/w/index.php?title=Windows_Defender&oldid=818651625 '' Categories : Microsoft software Microsoft Windows security technology Windows - only
freeware Spyware removal Windows components 2006 software Windows security software Hidden categories : CS1 German - language sources ( de ) CS1 Japanese -
language sources ( ja ) Articles needing additional references from July 2009 All articles needing additional references Talk Search Contents About Wikipedia
Čeština Dansk Deutsch Ελληνικά Español فارسی Français 한국어 Bahasa Indonesia Italiano עברית Bahasa Melayu Nederlands 日本 語 Norsk Polski Português Română Русский
සිංහල Simple English Suomi Svenska Tagalog தமிழ் ไทย Türkçe Українська Tiếng Việt 中文 Edit links This page was last edited on 4 January 2018 , at 20 : 09 . Text
is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply .
</item 8>

<item 9>
Windows Defender
Since Windows 10 Anniversary Update , offline functionality is integrated into the regular Windows Defender program . Mitigated Security vulnerability ( edit )
On 5 May 2017 , Tavis Ormandy , a vulnerability researcher for Google , discovered a security vulnerability in the JavaScript analysis module ( NScript ) of
Microsoft Antimalware Engine ( MsMpEngine ) that impacted Windows Defender , Microsoft Security Essentials and System Center Endpoint Protection . By 8 May 2017
, Microsoft had released a patch to all affected systems . Ars Technica commended Microsoft for its unprecedented patching speed and said that the disaster had
been averted . See also ( edit ) Security and safety features new to Windows Vista References ( edit ) Jump up ^ `` Windows Defender '' . Download Center .
Microsoft . 23 May 2007 . Archived from the original on 29 April 2012 . ^ Jump up to : `` What is Windows Defender ? '' . Computer Hope . Retrieved 31 December
2017 . Jump up ^ Shultz , Greg . `` Windows Defender : Past , present , and future '' . Retrieved 13 June 2017 .
</item 9>

<item 10>
Windows Defender
The same update introduced a new flat logo . History ( edit ) Beta ( edit ) Microsoft AntiSpyware Beta 1 ( Version 1.0. 701 ) Windows Defender is based on GIANT
AntiSpyware , which was originally developed by GIANT Company Software , Inc . The company 's acquisition was announced by Microsoft on December 16 , 2004 .
While the original GIANT AntiSpyware supported older Windows versions , support for the Windows 9x line of operating systems was later dropped . The first beta
version of Microsoft AntiSpyware was released on January 6 , 2005 and was basically a repackaged GIANT AntiSpyware . More builds were released in 2005 , with
the last Beta 1 refresh released on November 21 , 2005 . At the 2005 RSA Security conference , Chief Software Architect and co-founder of Microsoft , Bill Gates
, announced that Windows Defender ( which was known as Microsoft AntiSpyware prior to November 4 , 2005 ) would be made available free of charge to all validly
licensed Windows 2000 , Windows XP , and Windows Server 2003 users to help secure their systems against the increasing malware threat .
</item 10>
=== END ITEM id=d32c30ef2c19a288 ===

=== ITEM id=0ca53df643067e54 ===
Gold sentences and facts:
Sentence 0: Duncan 's diagnosis was publicly confirmed during a CDC news conference the same day . That evening , Duncan reported feeling better and requested to watch a movie . The following morning , Duncan was breathing rapidly and complaining of `` pain all over '' . By the afternoon , however , he was able to eat , and the doctor noted that he was feeling better . The next day , October 3 , Duncan again reported feeling abdominal pain . That evening , the hospital contacted Chimerix , a biotechnology company developing Brincidofovir to combat the disease . The next day , Duncan 's organs were failing , and he was intubated to help him breathe . In the afternoon , the hospital began administering Brincidofovir . Nurses Nina Pham and Amber Joy Vinson continued to care for Duncan around the clock . On October 7 , the hospital reported that Duncan 's condition was improving . However , Duncan died at 7 : 51 a.m. on October 8 , becoming the first person to die in the United States of Ebola virus disease and the index patient for the later infections of nurses Pham and Vinson .
  Fact 0.0: The CDC confirmed Duncan's Ebola diagnosis at a news conference on the same day.
  Fact 0.1: That evening, Duncan reported feeling better and requested to watch a movie.
  Fact 0.2: The following morning, Duncan was breathing rapidly and complaining of 'pain all over'.
  Fact 0.3: By the afternoon, Duncan was able to eat.
  Fact 0.4: By the afternoon, the doctor noted that Duncan was feeling better.
  Fact 0.5: On October 3, Duncan again reported feeling abdominal pain.
  Fact 0.6: That evening, October 3, the hospital contacted Chimerix.
  Fact 0.7: Chimerix is a biotechnology company developing Brincidofovir to combat the disease.
  Fact 0.8: On October 4, Duncan's organs were failing.
  Fact 0.9: On October 4, Duncan was intubated to help him breathe.
  Fact 0.10: On October 4 in the afternoon, the hospital began administering Brincidofovir to Duncan.
  Fact 0.11: Nurses Nina Pham and Amber Joy Vinson continued to care for Duncan around the clock.
  Fact 0.12: On October 7, the hospital reported that Duncan's condition was improving.
  Fact 0.13: Duncan died at 7:51 a.m. on October 8.
  Fact 0.14: Duncan became the first person to die in the United States of Ebola virus disease.
  Fact 0.15: Duncan was the index patient for the later infections of nurses Pham and Vinson.
Sentence 1: In December 2014 , Ebola virus cases in the United States occurred due to four laboratory - confirmed cases of Ebola virus disease ( commonly known as `` Ebola '' ) in the United States . Eleven cases have been reported , including these four cases and seven cases medically evacuated from other countries ; the first was reported in September 2014 . Nine of the people contracted the disease outside the US and traveled into the country , either as regular airline passengers or as medical evacuees ; of those nine , two died . Two people have contracted Ebola in the United States . Both were nurses who treated an Ebola patient ; both have recovered .
  Fact 1.0: In December 2014, Ebola virus cases occurred in the United States due to four laboratory-confirmed cases of Ebola virus disease in the United States.
  Fact 1.1: Ebola virus disease is commonly known as Ebola.
  Fact 1.2: Eleven Ebola cases have been reported.
  Fact 1.3: The eleven reported cases include the four cases diagnosed in the United States and seven cases medically evacuated from other countries.
  Fact 1.4: The first of the eleven cases was reported in September 2014.
  Fact 1.5: Nine of the eleven cases contracted the disease outside the US and traveled into the country.
  Fact 1.6: The nine people traveled into the country either as regular airline passengers or as medical evacuees.
  Fact 1.7: Two of those nine people died.
  Fact 1.8: Two people have contracted Ebola in the United States.
  Fact 1.9: Both people who contracted Ebola in the United States were nurses who treated an Ebola patient.
  Fact 1.10: Both nurses have recovered.
Sentence 2: On September 30 , 2014 , the CDC announced that Thomas Eric Duncan , a reportedly 42 - year - old ( later corrected by CDC reports as a 45 - year - old ) Liberian national visiting the United States from Liberia , had been diagnosed with Ebola in Dallas , Texas . Duncan , who had been visiting family in Dallas , was treated at Texas Health Presbyterian Hospital Dallas . By October 4 , Duncan 's condition had deteriorated from `` serious but stable '' to `` critical '' . On October 8 , Duncan died of Ebola. The other three cases diagnosed in the United States as of October 2014 were :
  Fact 2.0: The CDC announced on September 30, 2014 that Thomas Eric Duncan had been diagnosed with Ebola in Dallas, Texas.
  Fact 2.1: Thomas Eric Duncan was reportedly 42 years old.
  Fact 2.2: CDC reports later corrected Thomas Eric Duncan's age to 45 years old.
  Fact 2.3: Thomas Eric Duncan was a Liberian national.
  Fact 2.4: Thomas Eric Duncan was visiting the United States from Liberia.
  Fact 2.5: Thomas Eric Duncan had been visiting family in Dallas.
  Fact 2.6: Thomas Eric Duncan was treated at Texas Health Presbyterian Hospital Dallas.
  Fact 2.7: By October 4, Thomas Eric Duncan's condition had deteriorated from 'serious but stable' to 'critical'.
  Fact 2.8: Thomas Eric Duncan died of Ebola on October 8.
  Fact 2.9: Three other cases were diagnosed in the United States as of October 2014, and they are listed after this sentence.
Sentence 3: Ebola virus disease in the U.S. Map of Ebola cases and infrastructure throughout the U.S. Cases contracted in the U.S. Cases first diagnosed in U.S. Cases evacuated to U.S. from other countries 7 Total cases 11 Deaths Recoveries from Ebola 9 Active cases 0
  Fact 3.0: The page is about Ebola virus disease in the U.S.
  Fact 3.1: The page includes a map of Ebola cases and infrastructure throughout the U.S.
  Fact 3.2: The page has categories for cases contracted in the U.S., cases first diagnosed in the U.S., and cases evacuated to the U.S. from other countries.
  Fact 3.3: The table lists 7 cases evacuated to the U.S. from other countries.
  Fact 3.4: The table lists 11 total cases.
  Fact 3.5: The table has a 'Deaths' label.
  Fact 3.6: The table lists 9 recoveries from Ebola.
  Fact 3.7: The table lists 0 active cases.

Retrieved items:
<item 1>
Ebola virus cases in the United States
CBS News . November 13 , 2014 . Jump up ^ Sieff , Kevin ( November 17 , 2014 ) . `` A doctor 's mistaken Ebola test : ' We were celebrating ... Then everything
fell apart ' '' . Washington Post . Jump up ^ `` Martin Salia , Surgeon With Ebola , Arrives in Nebraska From Sierra Leone '' . New York Times . November 15 ,
2014 . Retrieved November 16 , 2014 . Jump up ^ Goodnough , Abby ; Trenchard , Tommy ( November 17 , 2014 ) . `` Doctor Being Treated for Ebola in Omaha Dies ''
. The New York Times . Jump up ^ U.S.
</item 1>

<item 2>
Starting in September 2014, the United States reported 11 Ebola cases: four diagnosed domestically and seven patients medically evacuated from other countries.
Nine patients recovered, two died, and no cases remained active.
</item 2>

<item 3>
In 2014, Thomas Eric Duncan contracted Ebola in Liberia, became the first case diagnosed in the U.S., and died in Dallas on October 8 after an initial hospital
visit and release. Nurses Nina Pham and Amber Vinson, infected while treating him, and New York doctor Craig Spencer recovered; these were the only locally
transmitted cases. Overall, 11 cases were reported, including seven medical evacuations; two patients died. Coverage also addressed contact monitoring,
treatment, racial-bias allegations,
</item 3>

<item 4>
During the 2014 Ebola outbreak, 11 cases were reported in the U.S., including seven patients medically evacuated from West Africa; two patients died. Thomas
Eric Duncan, the first U.S.-diagnosed case, died in Dallas after initially being discharged, and nurses Nina Pham and Amber Vinson recovered after becoming
infected while treating him. Craig Spencer also recovered. Vinson’s travel prompted quarantines and scrutiny of screening. The response drew criticism over CDC
guidance, protective equipment, training
</item 4>

<item 5>
In 2014, nine people who contracted Ebola abroad entered the United States; two died. Four cases were diagnosed in the country: Liberian visitor Thomas Eric
Duncan, who became ill in Dallas and died October 8, and nurses Nina Pham and Amber Vinson, who caught Ebola while treating him and recovered, plus physician
Craig Spencer, infected while working in Guinea, who recovered. The nurses were the only locally transmitted cases; no one contracted Ebola in the U.S. and
died. Hundreds
</item 5>

<item 6>
Ebola virus cases in the United States
Ebola virus cases in the United States - wikipedia Ebola virus cases in the United States Jump to : navigation , search This article is about Ebola virus cases
in the United States starting in 2014 . For the 2013 - 2016 outbreak in Africa , see Ebola virus epidemic in West Africa . For the U.S. discovery of a different
species of the Ebolavirus genus in 1989 , see Reston virus . Ebola virus disease in the U.S. Map of Ebola cases and infrastructure throughout the U.S. Cases
contracted in the U.S. Cases first diagnosed in U.S. Cases evacuated to U.S. from other countries 7 Total cases 11 Deaths Recoveries from Ebola 9 Active cases 0
In December 2014 , Ebola virus cases in the United States occurred due to four laboratory - confirmed cases of Ebola virus disease ( commonly known as `` Ebola
'' ) in the United States . Eleven cases have been reported , including these four cases and seven cases medically evacuated from other countries ; the first
was reported in September 2014 .
</item 6>

<item 7>
Ebola virus cases in the United States
Nine of the people contracted the disease outside the US and traveled into the country , either as regular airline passengers or as medical evacuees ; of those
nine , two died . Two people have contracted Ebola in the United States . Both were nurses who treated an Ebola patient ; both have recovered . On September 30
, 2014 , the CDC announced that Thomas Eric Duncan , a reportedly 42 - year - old ( later corrected by CDC reports as a 45 - year - old ) Liberian national
visiting the United States from Liberia , had been diagnosed with Ebola in Dallas , Texas . Duncan , who had been visiting family in Dallas , was treated at
Texas Health Presbyterian Hospital Dallas . By October 4 , Duncan 's condition had deteriorated from `` serious but stable '' to `` critical '' . On October 8 ,
Duncan died of Ebola. The other three cases diagnosed in the United States as of October 2014 were : October 11 , 2014 , a nurse , Nina Pham , who had provided
care to Duncan at the hospital . October 14 , 2014 , Amber Joy Vinson , another nurse who treated Duncan .
</item 7>

<item 8>
Ebola virus cases in the United States
See also ( edit ) Ebola virus epidemic in Guinea Ebola virus epidemic in Liberia Ebola virus epidemic in Nigeria Ebola virus epidemic in Sierra Leone Ebola
virus disease in Spain List of Ebola outbreaks West Nile virus in the United States Notes ( edit ) Jump up ^ Duncan contracted Ebola in Liberia and was
diagnosed in the U.S. Two of the victims had had direct contact with Duncan . One victim was diagnosed in New York City ; his case is unrelated to the Texas
cases . ^ Jump up to : This takes into account the seven Americans who were medically evacuated from West Africa with Ebola . References ( edit ) Jump up ^ ``
Who are the American Ebola patients ? '' , CNN , October 6 , 2014 Jump up ^ `` Nebraska Hospital : Surgeon With Ebola Has Died '' . ABC News . November 17 ,
2014 . Retrieved November 17 , 2014 . Jump up ^ `` 2014 Ebola Outbreak - Case Count '' . CDC . December 15 , 2014 . Retrieved December 16 , 2014 . Jump up ^ ``
First diagnosed case of Ebola in the U.S. '' CNN .
</item 8>

<item 9>
Ebola virus cases in the United States
Medical evacuations from West Africa ( edit ) Local transmissions -- no deaths : Texas Initial cases -- deaths : Texas Initial cases -- no deaths : New York
Medically evacuated cases -- no deaths : Georgia , Maryland Medically evacuated cases -- deaths : Nebraska By the time the Ebola virus epidemic in West Africa
was over in June 2016 , seven Americans had been evacuated to the U.S. for treatment after contracting Ebola virus while working with medical teams trying to
stop the epidemic . Kent Brantly , a physician and medical director in Liberia for the aid group Samaritan 's Purse , and co-worker Nancy Writebol were infected
in July 2014 , while working in Monrovia . Both were flown to the United States at the beginning of August via the CDC 's Aeromedical Biological Containment
System for further treatment in Atlanta 's Emory University Hospital . On August 21 , Brantly and Writebol recovered and were discharged . Brantly has since
donated blood to three others with Ebola ( Sacra , Mukpo and Pham ) .
</item 9>

<item 10>
Ebola virus cases in the United States
In addition , on September 21 , a CDC employee , who showed no symptoms of the disease , was flown back to the United States as a preventative measure after low
- risk exposure with a healthcare worker ; he posed no risks to his family or the United States , and he is not known to have died . On September 28 , a fourth
American doctor was admitted to National Institutes of Health hospital . On October 2 , NBC News photojournalist Ashoka Mukpo , covering the outbreak in Liberia
, tested positive for Ebola after showing symptoms . Four other members of the NBC team , including physician Nancy Snyderman , were being closely monitored for
symptoms . Mukpo was evacuated on October 6 to the University of Nebraska Medical Center for treatment in their isolation unit . On October 21 , Mukpo was
declared Ebola - free and allowed to return to his home in Rhode Island . A government official in Sierra Leone announced on November 13 , that a doctor from
Sierra Leone , a permanent resident of the United States married to a U.S. citizen , would be transported to the Nebraska Medical Center for treatment for Ebola
. The doctor , identified later as Dr.
</item 10>
=== END ITEM id=0ca53df643067e54 ===

=== ITEM id=356823ab5e91e517 ===
Gold sentences and facts:
Sentence 0: Deion Sanders Sanders in 2011 No. 21 , 37 Position : Cornerback Date of birth : ( 1967 - 08 - 09 ) August 9 , 1967 ( age 50 ) Place of birth : Fort Myers , Florida Height : 6 ft 1 in ( 1.85 m ) Weight : 198 lb ( 90 kg ) Career information High school : North Fort Myers ( FL ) College : Florida State NFL Draft : 1989 / Round : 1 / Pick : 5 Career history Atlanta Falcons ( 1989 -- 1993 ) San Francisco 49ers ( 1994 ) Dallas Cowboys ( 1995 -- 1999 ) Washington Redskins ( 2000 ) Baltimore Ravens ( 2004 -- 2005 ) Career highlights and awards 2 × Super Bowl champion ( XXIX , XXX ) 8 × Pro Bowl ( 1991 -- 1994 , 1996 -- 1999 ) 8 × First - team All - Pro ( 1991 -- 1994 , 1996 -- 1999 ) NFL Defensive Player of the Year ( 1994 ) NFL 1990s All - Decade Team Atlanta Falcons Ring of Honor Jim Thorpe Award ( 1988 ) 2 × Unanimous All - American ( 1987 , 1988 ) Florida State Seminoles No. 2 retired Career NFL statistics Interceptions : 53 Defensive touchdowns : 10 Return touchdowns : 9 Receptions : 60 Receiving yards : 784 Receiving touchdowns : 3 Player stats at NFL.com Player stats at PFR Pro Football Hall of Fame College Football Hall of Fame
  Fact 0.0: Deion Sanders is pictured in 2011.
  Fact 0.1: Deion Sanders' jersey numbers are No. 21 and No. 37.
  Fact 0.2: Deion Sanders' position is cornerback.
  Fact 0.3: Deion Sanders was born on August 9, 1967.
  Fact 0.4: Deion Sanders is listed as age 50.
  Fact 0.5: Deion Sanders was born in Fort Myers, Florida.
  Fact 0.6: Deion Sanders is 6 ft 1 in (1.85 m) tall.
  Fact 0.7: Deion Sanders weighs 198 lb (90 kg).
  Fact 0.8: Deion Sanders' high school is North Fort Myers (FL).
  Fact 0.9: Deion Sanders' college is Florida State.
  Fact 0.10: Deion Sanders was picked in round 1, pick 5 of the 1989 NFL Draft.
  Fact 0.11: Deion Sanders played for the Atlanta Falcons from 1989 to 1993.
  Fact 0.12: Deion Sanders played for the San Francisco 49ers in 1994.
  Fact 0.13: Deion Sanders played for the Dallas Cowboys from 1995 to 1999.
  Fact 0.14: Deion Sanders played for the Washington Redskins in 2000.
  Fact 0.15: Deion Sanders played for the Baltimore Ravens from 2004 to 2005.
  Fact 0.16: Deion Sanders is a 2-time Super Bowl champion, for Super Bowls XXIX and XXX.
  Fact 0.17: Deion Sanders is an 8-time Pro Bowl selection (1991-1994, 1996-1999).
  Fact 0.18: Deion Sanders is an 8-time First-team All-Pro (1991-1994, 1996-1999).
  Fact 0.19: Deion Sanders was NFL Defensive Player of the Year in 1994.
  Fact 0.20: Deion Sanders is named to the NFL 1990s All-Decade Team.
  Fact 0.21: Deion Sanders has an Atlanta Falcons Ring of Honor entry.
  Fact 0.22: Deion Sanders won the Jim Thorpe Award in 1988.
  Fact 0.23: Deion Sanders is a 2-time Unanimous All-American (1987, 1988).
  Fact 0.24: No. 2 is retired by the Florida State Seminoles.
  Fact 0.25: Deion Sanders' career NFL interceptions total 53.
  Fact 0.26: Deion Sanders' career NFL defensive touchdowns total 10.
  Fact 0.27: Deion Sanders' career NFL return touchdowns total 9.
  Fact 0.28: Deion Sanders' career NFL receptions total 60.
  Fact 0.29: Deion Sanders' career NFL receiving yards total 784.
  Fact 0.30: Deion Sanders' career NFL receiving touchdowns total 3.
  Fact 0.31: The page links to player stats at NFL.com.
  Fact 0.32: The page links to player stats at PFR.
  Fact 0.33: The page links to the Pro Football Hall of Fame.
  Fact 0.34: The page links to the College Football Hall of Fame.

Retrieved items:
<item 1>
Deion Sanders
1996 -- 1999 ) NFL Defensive Player of the Year ( 1994 ) NFL 1990s All - Decade Team Atlanta Falcons Ring of Honor Jim Thorpe Award ( 1988 ) 2 × Unanimous All -
American ( 1987 , 1988 ) Florida State Seminoles No. 2 retired Career NFL statistics Interceptions : 53 Defensive touchdowns : 10 Return touchdowns : 9
Receptions : 60 Receiving yards : 784 Receiving touchdowns : 3 Player stats at NFL.com Player stats at PFR Pro Football Hall of Fame College Football Hall of
Fame Deion Luywnn Sanders Sr. ( / ˈdiːɒn / ; born August 9 , 1967 ) , nicknamed Primetime , is a former American football and baseball player who works as an
analyst for CBS Sports and the NFL Network . He was inducted into the Pro Football Hall of Fame in 2011 . Sanders played football primarily at cornerback , but
also as a kick returner , punt returner and occasionally wide receiver .
</item 1>

<item 2>
Deion Sanders
Deion Sanders - wikipedia Deion Sanders Jump to : navigation , search Deion Sanders Sanders in 2011 No. 21 , 37 Position : Cornerback Date of birth : ( 1967 -
08 - 09 ) August 9 , 1967 ( age 50 ) Place of birth : Fort Myers , Florida Height : 6 ft 1 in ( 1.85 m ) Weight : 198 lb ( 90 kg ) Career information High
school : North Fort Myers ( FL ) College : Florida State NFL Draft : 1989 / Round : 1 / Pick : 5 Career history Atlanta Falcons ( 1989 -- 1993 ) San Francisco
49ers ( 1994 ) Dallas Cowboys ( 1995 -- 1999 ) Washington Redskins ( 2000 ) Baltimore Ravens ( 2004 -- 2005 ) Career highlights and awards 2 × Super Bowl
champion ( XXIX , XXX ) 8 × Pro Bowl ( 1991 -- 1994 , 1996 -- 1999 ) 8 × First - team All - Pro ( 1991 -- 1994 ,
</item 2>

<item 3>
Deion Sanders
On February 5 , 2011 , Sanders was announced as a Pro Football Hall of Fame inductee in his first year of eligibility . On May 17 , 2011 , Sanders was announced
as a College Football Hall of Fame inductee . On August 6 , 2011 , Sanders was inducted into the Pro Football Hall of Fame . Sanders also had a rushing TD in
the playoffs ( against the Philadelphia Eagles in January 1996 ) . This makes him ( including post season ) one of only two players in NFL history ( Bill Dudley
being the other ) to score a touchdown six different ways ( interception return , punt return , kickoff return , receiving , rushing , and a fumble recovery ) .
On February 6 , 2011 , at Super Bowl XLV in Arlington , Texas , Sanders performed the pre-game coin toss . Sanders did not attend classes or take final exams
during the fall semester of his senior year at Florida State , yet played in the Sugar Bowl against Auburn . This caused the Florida State Legislature to create
the ' Deion Sanders rule ' , whereby a football athlete at any state university could not play in a bowl game if he failed to successfully complete the previous
semester .
</item 3>

<item 4>
Deion Sanders
WR Wendell Davis TE Keith Jackson OL Mark Hutson OL Dave Cadigan OL John Elliott OL Randall McDaniel C Nacho Albergamo Defense DL Daniel Stubbs DL Chad Hennings
DL Tracy Rocker DL Ted Gregory DL John Roper LB Chris Spielman LB Aundray Bruce LB Dante Jones DB Bennie Blades DB Deion Sanders DB Rickey Dixon DB Chuck Cecil
Special teams PK David Treadwell P Tom Tupa 1988 College Football All - America Team consensus selections Offense QB Steve Walsh & Troy Aikman RB Barry Sanders
RB Anthony Thompson RB Tim Worley WR Jason Phillips WR Hart Lee Dykes TE Marv Cook OL Tony Mandarich OL Anthony Phillips OL Mike Utley OL Mark Stepnoski C Jake
Young C John Vitale Defense DL Mark Messner DL Tracy Rocker DL Wayne Martin DL Frank Stams DL Bill Hawkins LB Derrick Thomas LB Broderick Thomas LB Mike
Stonebreaker DB Deion Sanders DB Donnell Woolford DB Louis Oliver DB Darryl Henley Special teams PK Kendall Trainor P Keith English Jim Thorpe Award winners
1986 : Everett 1987 : Blades & Dixon 1988 : Sanders 1989 : Carrier 1990 : Lewis 1991 : Buckley
</item 4>

<item 5>
Deion Sanders
NFL . National Football League . Retrieved 28 January 2017 . Jump up ^ `` Deion Sanders HOF Speech '' . Video Entry . Retrieved October 19 , 2011 . Jump up ^
Kallestad , Brent . `` Deion Changes the Rules Again : Pro football : This time , however , it 's for better education of athletes with new standards in Florida
. '' . Los Angeles Times . Retrieved 29 January 2017 . Jump up ^ Trex , Ethan . `` Two Rules Named After Deion Sanders '' . mental_floss . Mental Floss , Inc .
Retrieved 29 January 2017 . Jump up ^ `` NFL Videos : 2011 HOF : Deion Sanders '' . Nfl.com . August 6 , 2011 . Retrieved August 16 , 2012 . Jump up ^ Frohman ,
Jimmy . `` 11 Most Awesome NFL Player Nicknames '' . Phactual . Phactual , LLC . Retrieved 24 January 2017 . Jump up ^ `` CES ProNews Flashes ! '' .
</item 5>

<item 6>
Deion Sanders
Sanders amassed 7,838 all - purpose yards and scored 22 touchdowns : nine interception returns , six punt returns , three kickoff returns , three receiving ,
and one fumble recovery . His 19 defensive and return touchdowns was an NFL record ( now held by Devin Hester ) . In the postseason , Sanders added 5 more
interceptions , as well as 3 receptions for 95 yards , 4 carries for 39 yards , and two touchdowns ( one rushing and one receiving ) . He was selected to eight
Pro Bowls and won the NFL Defensive Player of the Year Award in 1994 . College Football News named Sanders # 8 in its list of 100 Greatest College Football
Players of All - Time . The Sporting News named Sanders # 37 in their Top 100 Football Players of the Century released in 1999 . ESPN named Sanders # 74 in its
list of the 100 Great Athletes of the Century released in 1999 . NFL.com named Sanders # 34 on NFL 's Top 100 list released in late 2010 On November 11 , 2010 ,
Sanders was inducted into the Atlanta Falcons ' Ring of Honor .
</item 6>

<item 7>
Deion Sanders
Sanders also occasionally lined up with his team 's offense . During the 1996 season , Sanders skipped the baseball season , concentrating on football , and
attended the first NFL training camp of his career to better familiarize himself with the nuances of the wide receiver position . He became only the second two
- way starter ( after the Cardinals ' Roy Green ) in the NFL since Chuck Bednarik for the first half of the season due . Sanders is the only man to play in both
a Super Bowl and a World Series , to hit an MLB home run and score an NFL touchdown in the same week , and to have both a reception and an interception in the
Super Bowl . He is one of two players to score an NFL touchdown six different ways . During his career , Sanders intercepted 53 passes for 1,331 yards ( a 25.1
yards per return average ) , recovered four fumbles for 15 yards , returned 155 kickoffs for 3,523 yards , gained 2,199 yards on 212 punt returns , and caught
60 passes for 784 yards .
</item 7>

<item 8>
Deion Sanders
In 2014 , he debuted his new show , the Deion Family Playbook . Contents ( hide ) 1 Early life 1.1 High school 1.2 College career 2 Professional baseball career
3 Professional football career 3.1 Draft and Atlanta Falcons 3.2 San Francisco 49ers 3.3 Dallas Cowboys 3.4 Washington Redskins 3.5 Baltimore Ravens 4 Legacy 5
Media appearances and pop culture fame 6 Other business and entertainment ventures 7 Discography 8 Personal life 9 See also 10 References 11 External links
Early life ( edit ) High school ( edit ) Sanders was born in Fort Myers , Florida . He attended North Fort Myers High School in North Fort Myers , and was a
letterman and All - State honoree in football , basketball and baseball . In 1985 , Sanders was named to the Florida High School Association All - Century Team
which selected the top 33 players in the 100 - year history of high school football in the state . The Kansas City Royals selected Sanders out of North Fort
Myers High School in the sixth round of the 1985 Major League Baseball draft . He did not sign with the Royals .
</item 8>

<item 9>
Deion Sanders
In 1995 , he signed with the Dallas Cowboys for a minimum yearly base salary and a nearly $13 million signing bonus in an attempt to circumvent the NFL 's
salary cap . This caused the NFL to institute its own ' Deion Sanders rule ' whereby a prorated portion of a player 's signing bonus counted against the salary
cap . Media appearances and pop culture Fame ( edit ) See also : Leon Sandcastle Sanders became known for sporting a `` do - rag '' or bandana and for his high
- stepping into the end zone followed by his touchdown dance celebrations . At the end of his Hall of Fame speech , he put a bandana on his bust . His `` Prime
Time '' nickname was given to him by a friend and high - school teammate , Florida Gators defensive back Richard Fain . The two played pickup basketball games
together during the prime time television hour , and Sanders ' athletic display during those games earned him the nickname . His other nickname is `` Neon Deion
'' . Following his first Super Bowl victory with the San Francisco 49ers , Sanders hosted Saturday Night Live , broadcast on February 18 , 1995 .
</item 9>

<item 10>
Deion Sanders
After getting into a scuffle with his former Falcon teammate Andre Rison , Sanders intercepted a pass from quarterback Jeff George and returned it 93 yards
while mockingly staring down the entire Falcons sideline before high - stepping into the end zone . Sanders was later voted the 1994 NFL Defensive Player of the
Year and recorded an interception in the 49ers 49 -- 26 victory over the San Diego Chargers in Super Bowl XXIX , earning him his first championship ring .
Dallas Cowboys ( edit ) Following his successful season with the 49ers , Sanders , along with his agent Eugene Parker , courted numerous teams in need of a
cornerback . The several teams in the `` Deion Sweepstakes '' , as it was called by the media , were the Philadelphia Eagles , Oakland Raiders , Miami Dolphins
, New Orleans Saints , San Francisco 49ers and the Dallas Cowboys , who had lost their starting cornerback Kevin Smith to injury for the rest of the season .
</item 10>
=== END ITEM id=356823ab5e91e517 ===
