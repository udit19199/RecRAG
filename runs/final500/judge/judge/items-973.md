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

=== ITEM id=1453f0628fe225d4 ===
Gold sentences and facts:
Sentence 0: The time that has elapsed since the Camp David Accords has left no doubt as to their enormous ramifications on Middle Eastern politics . Most notably , the perception of Egypt within the Arab world changed . With the most powerful of the Arab militaries and a history of leadership in the Arab world under Nasser , Egypt had more leverage than any of the other Arab states to advance Arab interests . Egypt was subsequently suspended from the Arab League from 1979 until 1989 .
  Fact 0.0: Time has elapsed since the Camp David Accords.
  Fact 0.1: The passage of time has left no doubt about the enormous ramifications of the Camp David Accords on Middle Eastern politics.
  Fact 0.2: The perception of Egypt within the Arab world changed.
  Fact 0.3: The change in perception of Egypt was the most notable effect.
  Fact 0.4: Egypt had the most powerful of the Arab militaries.
  Fact 0.5: Egypt had a history of leadership in the Arab world under Nasser.
  Fact 0.6: Egypt had more leverage than any other Arab state to advance Arab interests.
  Fact 0.7: Egypt was subsequently suspended from the Arab League.
  Fact 0.8: Egypt's suspension from the Arab League lasted from 1979 until 1989.

Retrieved items:
<item 1>
Camp David Accords
According to The Continuum Political Encyclopedia of the Middle East : The normalization of relations ( between Israel and Egypt ) went into effect in January
1980 . Ambassadors were exchanged in February . The boycott laws were repealed by Egypt 's National Assembly the same month , and some trade began to develop ,
albeit less than Israel had hoped for . In March 1980 regular airline flights were inaugurated . Egypt also began supplying Israel with crude oil `` . According
to Kenneth Stein in Heroic Diplomacy : Sadat , Kissinger , Carter , Begin , and the Quest for Arab -- Israeli Peace : The Accords were another interim agreement
or step , but negotiations that flowed from the Accords slowed for several reasons . These included an inability to bring the Jordanians into the discussions ;
the controversy over settlements ; the inconclusive nature of the subsequent autonomy talks ; domestic opposition sustained by both Begin and Sadat and , in
Sadat 's case , ostracism and anger from the Arab world ; the emergence of a what became a cold peace between Egypt and Israel ; and changes in foreign policy
priorities including discontinuity in personnel committed to sustaining the negotiating process ( . )
</item 1>

<item 2>
Camp David Accords
Accompanied by their capable negotiating teams and with their respective interests in mind , both leaders converged on Camp David for 13 days of tense and
dramatic negotiations from 5 to 17 September 1978 . By all accounts , Carter 's relentless drive to achieve peace and his reluctance to allow the two men to
leave without reaching an agreement are what played the decisive role in the success of the talks . Carter 's advisers insisted on the establishment of an
Egyptian - Israeli agreement which would lead to an eventual solution to the Palestine issue . They believed in a short , loose , and overt linkage between the
two countries amplified by the establishment of a coherent basis for a settlement . However , Carter felt they were not `` aiming high enough '' and was
interested in the establishment of a written `` land for peace '' agreement with Israel returning the Sinai Peninsula and West Bank . Numerous times both the
Egyptian and Israeli leaders wanted to scrap negotiations , only to be lured back into the process by personal appeals from Carter .
</item 2>

<item 3>
Camp David Accords
With the withdrawal , Israel also returned Egypt 's Abu - Rudeis oil fields in western Sinai , which contained long term , commercially productive wells . The
agreement also resulted in the United States committing to several billion dollars worth of annual subsidies to the governments of both Israel and Egypt ,
subsidies which continue to this day , and are given as a mixture of grants and aid packages committed to purchasing U.S. materiel . From 1979 ( the year of the
peace agreement ) to 1997 , Egypt received military aid of US $1.3 billion annually , which also helped modernize the Egyptian military . ( This is beyond
economic , humanitarian , and other aid , which has totaled more than US $25 billion . ) Eastern - supplied until 1979 , Egypt now received American weaponry
such as the M1A1 Abrams Tank , AH - 64 Apache gunship and the F - 16 fighter jet . In comparison , Israel has received $3 billion annually since 1985 in grants
and military aid packages . Consequences The time that has elapsed since the Camp David Accords has left no doubt as to their enormous ramifications on Middle
Eastern politics .
</item 3>

<item 4>
Camp David Accords
Most notably , the perception of Egypt within the Arab world changed . With the most powerful of the Arab militaries and a history of leadership in the Arab
world under Nasser , Egypt had more leverage than any of the other Arab states to advance Arab interests . Egypt was subsequently suspended from the Arab League
from 1979 until 1989 . When the Camp David accords were signed , Jordan 's King Hussein saw it as a slap to the face when Sadat volunteered Jordan 's
participation in deciding how functional autonomy would work . More specifically , Sadat effectively said that Jordan would have a role in how the West Bank
would be administered . Like the Rabat Summit Resolution , the Camp David Accords circumscribed Jordan 's objective to reassert its control over the West Bank .
Focusing as it did on Egypt , the Carter administration accepted Sadat 's claim that he could deliver Hussein . However , with Arab world opposition building
against Sadat , Jordan could not risk accepting the Accords without the support from powerful Arab neighbours , like Iraq , Saudi Arabia , and Syria . Hussein
consequently felt diplomatically snubbed .
</item 4>

<item 5>
Camp David Accords
Meir Rosenne and Aharon Barak . Furthermore , the absence of the media contributed to the Accord 's successes : there were no possibilities provided to either
leader to reassure his political body or be driven to conclusions by members of his opposition . An eventual scrap of negotiations by either leader would have
proven disastrous , resulting in taking the blame for the summit 's failure as well as a disassociation from the White House . Ultimately , neither Begin nor
Sadat was willing to risk those eventualities . Both of them had invested enormous amounts of political capital and time to reach an agreement . Partial
agreements Egyptian President Anwar Sadat and Israeli Prime Minister Menachem Begin acknowledge applause during a joint session of Congress in Washington , D.C.
, during which President Jimmy Carter announced the results of the Camp David Accords , 18 September 1978 . `` Remarks on the Signing of the Camp David Accords
'' Play media Jimmy Carter , seated with Egyptian President Anwar Sadat and Israeli Prime Minister Menachem Begin , makes statements at a Joint session of the
United States Congress following the Camp David Accords . Problems playing this file ? See media help .
</item 5>

<item 6>
Camp David Accords
The issues of Israeli settlement withdrawal from the Sinai and the status of the West Bank created what seemed to be an impasse . In response , Carter had the
choice of trying to salvage the agreement by conceding the issue of the West Bank to Begin , while advocating Sadat 's less controversial position on the
removal of all settlements from the Sinai Peninsula . Or he could have refused to continue the talks , reported the reasons for their failure , and allowed
Begin to bear the brunt of the blame . Carter chose to continue and for three more days negotiated . During this time , Carter even took the two leaders to the
nearby Gettysburg National Military Park in the hopes of using the American Civil War as a simile to their own struggle . Consequently , the 13 days marking the
Camp David Accords were considered a success . Partly due to Carter 's determination in obtaining an Israeli - Egyptian agreement , a full two - week pledge to
a singular international problem . Additionally , Carter was beneficiary to a fully pledged American foreign team . Likewise , the Israeli delegation had a
stable of excellent talent in Ministers Dayan and Weizman and legal experts Dr.
</item 6>

<item 7>
Camp David Accords
Camp David Accords - wikipedia Camp David Accords Jump to : navigation , search Camp David Accords Framework for Peace in the Middle East and Framework for the
Conclusion of a Peace Treaty between Egypt and Israel Celebrating the signing of the Camp David Accords : Menachem Begin , Jimmy Carter , Anwar Sadat Type
Bilateral treaty Signed 17 September 1978 ( 1978 - 09 - 17 ) Location Washington , DC , United States Original signatories Egypt Israel Signatories Menachem
Begin ( Prime Minister of Israel ) Anwar Sadat ( President of Egypt ) Jimmy Carter ( President of the United States ) Ratifiers Egypt Israel Language English
French The Camp David Accords were signed by Egyptian President Anwar Sadat and Israeli Prime Minister Menachem Begin on 17 September 1978 , following twelve
days of secret negotiations at Camp David . The two framework agreements were signed at the White House , and were witnessed by United States President Jimmy
Carter . The second of these frameworks ( A Framework for the Conclusion of a Peace Treaty between Egypt and Israel ) led directly to the 1979 Egypt -- Israel
Peace Treaty .
</item 7>

<item 8>
Camp David Accords
Lastly , the biggest consequence of all may be in the psychology of the participants of the Arab - Israeli conflict . The success of Begin , Sadat , and Carter
at Camp David demonstrated to other Arab states and entities that negotiations with Israel were possible -- that progress results only from sustained efforts at
communication and cooperation . Despite the disappointing conclusion of the 1993 Oslo Accords between the PLO and Israel , and even though the 1994 Israel --
Jordan peace treaty has not fully normalized relations with Israel , both of these significant developments had little chance of occurring without the precedent
set by Camp David . Public support Although most Israelis supported the Accords , the Israeli settler movement opposed them because Sadat would not agree to a
treaty in which Israel had any presence in the Sinai Peninsula at all , Israel had to withdraw from the entire Sinai Peninsula . Israeli settlers tried to
prevent the government from dismantling their settlements . In Israel , there is lasting support of the Camp David Peace Accords , which have become a national
consensus , supported by 85 % of Israelis according to a 2001 poll taken by the Jaffee Center for Strategic Studies ( Israel based ) .
</item 8>

<item 9>
Camp David Accords
Hussein consequently felt diplomatically snubbed . One of Carter 's regrets was allowing Sadat to claim that he could speak for Hussein if Jordan refused to
join the talks , but by then the damage was done to the Jordanians . United States President Jimmy Carter greeting Egyptian President Anwar Sadat at the White
House shortly after the Camp David Accords went into effect , 8 April 1980 . The Camp David Accords also prompted the disintegration of a united Arab front in
opposition to Israel . Egypt 's realignment created a power vacuum that Saddam Hussein of Iraq , at one time only a secondary power , hoped to fill . Because of
the vague language concerning the implementation of Resolution 242 , the Palestinian problem became the primary issue in the Arab -- Israeli conflict
immediately following the Camp David Accords ( and , arguably , until today ) . Many of the Arab nations blamed Egypt for not putting enough pressure on Israel
to deal with the Palestinian problem in a way that would be satisfactory to them . Syria also informed Egypt that it would not reconcile with the nation unless
it abandoned the peace agreement with Israel .
</item 9>

<item 10>
Camp David Accords
The efforts initially focused on a comprehensive resolution of disputes between Israel and the Arab countries , gradually evolving into a search for a bilateral
agreement between Israel and Egypt . Upon assuming office on January 20 , 1977 , President Carter moved to rejuvenate the Middle East peace process that had
stalled throughout the 1976 presidential campaign in the United States . Following the advice of a Brookings Institution report , Carter opted to replace the
incremental , bilateral peace talks which had characterized Henry Kissinger 's shuttle diplomacy following the 1973 Yom Kippur War with a comprehensive ,
multilateral approach . The Yom Kippur War further complicated efforts to achieve the objectives written in United Nations Security Council Resolution 242 .
Israel 's Prime Minister Yitzhak Rabin and his successor , Menachem Begin , were both skeptical of an international conference . While Begin , who took office
in May 1977 , officially favored the reconvening of the conference , perhaps even more vocally than Rabin , and even accepted the Palestinian presence , in
actuality the Israelis and the Egyptians were secretly formulating a framework for bilateral talks .
</item 10>
=== END ITEM id=1453f0628fe225d4 ===

=== ITEM id=b227c77ed52e932b ===
Gold sentences and facts:
Sentence 0: When leaving work in Idaho 's Snake River Valley , Billy Colman sees a pack of dogs attacking a stray coonhound . He takes the stray home to feed it . Once it has rested , Billy sets it free , knowing that it will return home .
  Fact 0.0: Billy Colman leaves work in Idaho's Snake River Valley.
  Fact 0.1: Billy Colman sees a pack of dogs attacking a stray coonhound.
  Fact 0.2: Billy Colman takes the stray coonhound home to feed it.
  Fact 0.3: After the coonhound has rested, Billy Colman sets it free.
  Fact 0.4: Billy Colman knows that the coonhound will return home.

Retrieved items:
<item 1>
Where the Red Fern Grows
Billy 's grandfather enters him into a championship coon hunt , putting him against experienced hunters and the finest dogs in the country . Before it starts ,
he enters Little Ann into a conformation contest , where she wins the silver cup . On the fourth night of the hunt , Old Dan and Little Ann tree three raccoons
, making it to the final round . The sixth night , they tree one before a blizzard hits . Billy , his father , grandfather , and the judge lose sight of the
dogs . When they finally find them , Billy 's grandfather sprains his ankle . They build a fire , and when Billy 's father chops down a tree , three raccoons
jump out . Old Dan and Little Ann take two of them down , and chase the third to another tree . In the morning , the hunters find them covered with ice circling
the tree . That raccoon wins them the championship , gold cup , and $300 of jackpot money . One night Old Dan and Little Ann tree a mountain lion , and it
attacks . Billy enters the fight with his axe , hoping to save his dogs , but they end up having to save him .
</item 1>

<item 2>
Where the Red Fern Grows
Every year in Tahlequah Oklahoma there is a Red Fern festival . It is listed below in the links . See also ( edit ) List of Where the Red Fern Grows Characters
Footnotes ( edit ) Jump up ^ Where the Red Fern Grows Discussion Guide Scholastic.com ^ Jump up to : Where the Red Fern Grows ( 1974 ) on IMDb Jump up ^ IMDb :
Where the Red Fern Grows : Part Two ^ Jump up to : Wilson Rawls , Dreams Can Come True , audio recording with commentary by Jim Trelease ^ Jump up to :
http://www.trelease-on-reading.com/rawls.html Jump up ^
https://news.google.com/newspapers?nid=336&dat=19740216&id=bMdSAAAAIBAJ&sjid=AH4DAAAAIBAJ&pg=5192,3528282&hl=en Jump up ^ `` All - Time Bestselling Children 's
Books '' . PublishersWeekly.com . Retrieved 2017 - 01 - 29 .
</item 2>

<item 3>
Where the Red Fern Grows
Reception ( edit ) Although sales of the novel began slowly , by 1974 over 90,000 copies had been sold . In 2001 , Publishers Weekly estimated that it had sold
6,754,308 copies . Today Where the Red Fern Grows is required reading in many American schools . One critic said it will please adults as well as children . I
remember crying so much through this book , and even today I tear up thinking of Old Dan and Little Ann . I also loaned this to my ( then ) children 's
librarian , because the library copy was always out . I even marked the pages , `` Get out tissue here . '' -- DeAnn Okamura I love , love , love this book with
all my heart and soul . My fourth grade teacher read it to me eons ago , and I 've read it to two of my three boys . There 's something about weeping together
uncontrollably that builds a community of readers ... -- Tess Alfonsin There is a statue of Billy and his dogs at the Idaho Falls Public Library . Every year in
Tahlequah Oklahoma there is a Red Fern festival .
</item 3>

<item 4>
Where the Red Fern Grows
As he tries to call them off , they look at him sadly and he cuts the tree down , which takes him two days and costs him blistered hands . In the end , when he
's about to give up his effort , Billy offers a short prayer for strength to continue . A strong wind starts to blow and the tree falls . Old Dan and Little Ann
take the raccoon down . Billy , Old Dan , and Little Ann go hunting almost every night and become well - known in the Ozarks . Billy and his grandfather make a
bet with Rubin and Rainie Pritchard that Old Dan and Little Ann can tree the `` ghost coon '' . It leads them on a long , complicated hunt , and Rubin and
Rainie want to give up . When they finally have it treed , Billy refuses to kill it . Rubin then orders his dog , Old Blue , to attack Old Dan , who is smaller
; and Rubin starts to beat Billy . Little Ann helps Old Dan , and together they injure Old Blue ; Rubin runs to attack them with an axe , but he trips , falls
on it and kills himself .
</item 4>

<item 5>
Where the Red Fern Grows
Background ( edit ) In a talk given to a group of schoolteachers , Wilson Rawls related how he wrote the first version of the novel ( along with five full
novels , and hundreds of short stories and novelettes ) during the years that he worked on construction in Mexico and Idaho . He rolled the manuscripts up and
saved them in a trunk at his parents ' home . When he met his fiancée , Sophie , he did not want her to know about his failed dreams of becoming a writer , so
about a week before he got married he visited his parents and burned all his manuscripts . He then returned to Idaho and married Sophie . About three months
later , he confessed to his wife that he had burned all his manuscripts and had always dreamed of being a writer . She encouraged him to rewrite one of his
stories . He quit his job and wrote the novel in just three weeks . He said , `` I had it memorized . '' He would not let her read it until it was finished . He
said , `` I finished it on a Friday . I gave it to her Saturday morning and I went to town . I stayed in town all day .
</item 5>

<item 6>
Where the Red Fern Grows
Once it has rested , Billy sets it free , knowing that it will return home . The experience reminds Billy of his childhood in the Ozark Mountains of Oklahoma .
He wants a dog , and his parents offer to get him a mongrel from a neighbor . He wants two coonhounds , but his parents tell him they can not afford them .
Billy finds a magazine ad for a Kentucky kennel which breeds Redbone Coonhounds and sells them for $25 each . He goes to work performing odd jobs and saves the
$50 he needs with the help of his grandfather . Since it takes two years to do so , his grandfather writes to see if the kennel will honor the ad . They do ,
and the price has dropped so the two puppies only cost $40 . Billy 's dogs are delivered to the freight depot in Tahlequah . His family does not have
transportation , so he finds his own way to get to the depot and walks through the hills . He picks up his puppies , which are a male and a female . He buys a
gift for each member of his family with his extra $10 before heading back home .
</item 6>

<item 7>
Where the Red Fern Grows
On the way back home , he spends the night in Robber 's Cave in the Sparrow Hawk Mountains . There he builds a fire and plays with the puppies . While trying to
sleep , he hears a noise that he realizes is the cry of a mountain lion . In the morning , he continues on . He comes to a sycamore tree and sees the names Dan
and Ann carved inside a heart in the bark and decides to name the puppies Old Dan and Little Ann . To train Old Dan and Little Ann , Billy traps a raccoon with
the help of his grandfather and uses its pelt to teach them to hunt . During their training , their personalities become apparent : Old Dan is brave and strong
, while Little Ann is very intelligent . Both are very loyal to each other and to Billy . On the first night of hunting season , Billy takes Old Dan and Little
Ann out for their first hunt . He promises them that if they tree a raccoon , he will do the rest . They tree one in a large sycamore , which Billy had
previously nicknamed `` The Big Tree '' .
</item 7>

<item 8>
Where the Red Fern Grows
Where the Red Fern Grows - wikipedia Where the Red Fern Grows Jump to : navigation , search For the film adaptations , see Where the Red Fern Grows ( 1974 film
) and Where the Red Fern Grows ( 2003 film ) . Where the Red Fern Grows First edition hardback cover Author Wilson Rawls Country United States Language English
Genre Children 's novel Publisher Doubleday Publication date 1961 Media type Print ( Hardcover ) Pages 245 pp ISBN 0 - 440 - 22814 - X OCLC 39850615 Where the
Red Fern Grows is a 1961 children 's novel by Wilson Rawls about a boy who buys and trains two Redbone Coonhound hunting dogs . Contents ( hide ) 1 Plot summary
2 Films 3 Background 4 Reception 5 See also 6 Footnotes 7 External links Plot summary ( edit ) When leaving work in Idaho 's Snake River Valley , Billy Colman
sees a pack of dogs attacking a stray coonhound . He takes the stray home to feed it . Once it has rested , Billy sets it free , knowing that it will return
home .
</item 8>

<item 9>
Where the Red Fern Grows
Eventually , they kill the mountain lion , but Old Dan is badly wounded , and Billy finds his intestines in a bush . He dies late that night . Billy is
heartbroken , and Little Ann loses the will to live , stops eating , and dies of grief a few days later on Old Dan 's grave . Billy 's father tries to tell him
that it is all for the best , because with the money they received from winning the championship hunt , they can move to town . He goes to visit Old Dan and
Little Ann 's graves and finds a giant red fern between them . According to Native American legend , only an angel can plant one . He feels ready to move on
knowing that his dogs are always going to be remembered . Films ( edit ) The novel was made into a 1974 film starring Stewart Petersen , James Whitmore ,
Beverly Garland , and Jack Ging . It was followed by a sequel in 1992 , which starred Wilford Brimley , Chad McQueen , Lisa Whelchel , and Karen Carlson . The
film was remade in 2003 and starred Joseph Ashton , Dabney Coleman , Ned Beatty and Dave Matthews .
</item 9>

<item 10>
Graph query row:
last_name: Colman
</item 10>
=== END ITEM id=b227c77ed52e932b ===

=== ITEM id=b78232b1d1bb0457 ===
Gold sentences and facts:
Sentence 0: A cover of the song by Stacey Markus is used at the end of episode nine ( `` The Invitation '' ) of the 2015 Netflix original series Grace and Frankie , starring Jane Fonda and Lily Tomlin .
  Fact 0.0: A cover of a song by Stacey Markus is used at the end of episode nine, 'The Invitation', of Grace and Frankie.
  Fact 0.1: Grace and Frankie is a 2015 Netflix original series.
  Fact 0.2: Grace and Frankie stars Jane Fonda.
  Fact 0.3: Grace and Frankie stars Lily Tomlin.

Retrieved items:
<item 1>
Let My Love Open the Door
at 22 : 30 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the
Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 1>

<item 2>
Graph query row:
entity: The Invitation
relationship: RELATED_TO
related: Let My Love Open the Door
</item 2>

<item 3>
Let My Love Open the Door
Imdb.com. 2017 - 05 - 12 . Retrieved 2018 - 03 - 30 . External links ( edit ) Lyrics of this song at MetroLyrics ( hide ) Pete Townshend Discography Studio
albums Who Came First Rough Mix ( w / Ronnie Lane ) Empty Glass All the Best Cowboys Have Chinese Eyes White City : A Novel The Iron Man : A Musical
Psychoderelict Classic Quadrophenia Meher Baba tribute albums ( w / Ronnie Lane et al . ) Happy Birthday I Am With Love O ' Parvardigar Live albums Deep End
Live !
</item 3>

<item 4>
Let My Love Open the Door
The group Rogue Wave also covered it in `` Cover me '' released in 2017 . List of films , TV shows , and other media featuring the track ( edit ) The song has
been used frequently in film , most notably the comedy genre , often as trailer music for movies such as Jerry Maguire and How Do You Know , and then as end
credit song for Mr. Deeds in 2002 ; Jersey Girl and Along Came Polly , both in 2004 . It was featured in the following films and TV series : Look Who 's Talking
( 1989 ) Grosse Pointe Blank as `` E. Cola Mix '' ( 1997 ) Mr.
</item 4>

<item 5>
Graph query row:
entity: Stacey Markus
relationship: RELATED_TO
related: Let My Love Open the Door
</item 5>

<item 6>
Graph query row:
entity: The Invitation
relationship: PART_OF
related: Grace and Frankie
</item 6>

<item 7>
Let My Love Open the Door
It reached number five in Canada . Contents ( hide ) 1 Background 2 Chart performance 2.1 Weekly charts 2.2 Year - end charts 3 Personnel 4 Covers 5 List of
Films , TV Shows , and Other Media featuring the track 6 References 7 External links Background ( edit ) Although Townshend is a devotee to the religious guru
Meher Baba , he claimed in the liner notes of his Anthology CD that `` Jesus sings '' on the track . `` Let My Love Open the Door '' was released as the second
single from Empty Glass in Britain , where it was backed with the non-album tracks `` Classified '' and `` Greyhound Girl . '' The song was a minor British hit
, reaching # 46 . The song saw more success when it was released as the debut single from Empty Glass in America , where the song reached # 9 . It was Pete
Townshend 's only solo top 10 hit on the Billboard Hot 100 , but The Who 's song `` I Can See For Miles , '' which was written by Townshend , reached the same
position on the chart 13 years earlier .
</item 7>

<item 8>
Let My Love Open the Door
Let My Love Open the Door - wikipedia Let My Love Open the Door Jump to : navigation , search `` Let My Love Open the Door '' Single by Pete Townshend from the
album Empty Glass B - side `` Greyhound Girl '' Released June 4 , 1980 ( US ) Format 7 '' Recorded March 1980 Genre Art rock , new wave Length 2 : 44 Label Atco
Records Songwriter ( s ) Pete Townshend Producer ( s ) Pete Townshend ,
</item 8>

<item 9>
Let My Love Open the Door
A cover of the song by Stacey Markus is used at the end of episode nine ( `` The Invitation '' ) of the 2015 Netflix original series Grace and Frankie ,
starring Jane Fonda and Lily Tomlin . The song is featured in a Walmart Christmas holiday television commercial which started airing in early December 2017 .
Lethal Weapon ( TV series ) , season 2 , episode 9 `` Fools Rush In '' play the song performed by `` Rogue wave '' ( 2017 ) . References ( edit ) Jump up ^
Unterberger , Richie ( 2012 - 06 - 14 ) . `` Let My Love Open the Door '' . Rolling Stone . Retrieved 2016 - 10 - 16 . Jump up ^ `` Pete Townshend , ' Empty
Glass ' 100 Best Albums of the Eighties '' . Rolling Stone . 1989 - 11 - 16 . Retrieved 2016 - 10 - 16 . ^ Jump up to : Bac-lac.gc.ca ^ Jump up to : `` Let My
Love Open the Door '' . Songfacts.com .
</item 9>

<item 10>
Let My Love Open the Door
Billboard Hot 100 59 U.S. Cashbox 92 Personnel ( edit ) Pete Townshend : vocals , guitars , synths John `` Rabbit '' Bundrick : `` straight '' keyboards Simon
Phillips : drums Tony Butler : bass guitar Ted Jensen - mastering engineer Covers ( edit ) A version of this song by Luminate appears on The Covers EP . It was
also covered by M. Ward ( for the 2005 Starbucks compilation album , Sweetheart 2005 : Love Songs ) Christian rock group Audio Adrenaline covered this song on
their 1999 album Underdog . Natalie Imbruglia covered the song on her album Male released in 2015 . Great Big Sea covered the song on their 2012 album XX .
Roger Daltrey covered the song in 2015 for a campaign involving Teen Cancer America and First Citizens Bank . A full version was released in February 2016 ,
with all proceeds going to Teen Cancer America . The group Rogue Wave also covered it in `` Cover me '' released in 2017 .
</item 10>
=== END ITEM id=b78232b1d1bb0457 ===

=== ITEM id=2e73e2a88d938bf5 ===
Gold sentences and facts:
Sentence 0: The Country Music Hall of Fame and Museum in Nashville , Tennessee , is one of the world 's largest museums and research centers dedicated to the preservation and interpretation of American vernacular music . Chartered in 1964 , the museum has amassed one of the world 's most extensive musical collections .
  Fact 0.0: The Country Music Hall of Fame and Museum is in Nashville, Tennessee.
  Fact 0.1: The Country Music Hall of Fame and Museum is one of the world's largest museums dedicated to the preservation and interpretation of American vernacular music.
  Fact 0.2: The Country Music Hall of Fame and Museum is one of the world's largest research centers dedicated to the preservation and interpretation of American vernacular music.
  Fact 0.3: The Country Music Hall of Fame and Museum was chartered in 1964.
  Fact 0.4: The Country Music Hall of Fame and Museum has amassed one of the world's most extensive musical collections.
Sentence 1: Country Music Hall of Fame and Museum Established 1961 ( 1961 ) Location 222 5th Ave S Nashville , Tennessee 37203 Type Hall of fame Visitors 1.1 million in 2015 Director Kyle Young Website www.countrymusichalloffame.org
  Fact 1.0: The Country Music Hall of Fame and Museum was established in 1961.
  Fact 1.1: The Country Music Hall of Fame and Museum is located at 222 5th Ave S, Nashville, Tennessee 37203.
  Fact 1.2: The type of the Country Music Hall of Fame and Museum is Hall of fame.
  Fact 1.3: The Country Music Hall of Fame and Museum had 1.1 million visitors in 2015.
  Fact 1.4: The director of the Country Music Hall of Fame and Museum is Kyle Young.
  Fact 1.5: The website of the Country Music Hall of Fame and Museum is www.countrymusichalloffame.org.

Retrieved items:
<item 1>
Country Music Hall of Fame and Museum
In addition to the galleries , the museum has the 776 - seat CMA Theater , the Taylor Swift Education Center , and multi-purpose event rental spaces . Other
historic properties of the Country Music Hall of Fame and Museum include the letterpress operation Hatch Show Print ( located inside the museum ) and Historic
RCA Studio B ( located on Music Row ) , Nashville 's oldest surviving recording studio , where recordings by Elvis Presley , Dolly Parton , Waylon Jennings ,
and many others were made . The Country Music Hall of Fame and Museum has developed multiple platforms to make its collection accessible to a wider audience .
From weekly instrument demonstrations to its flagship songwriting program for schools , Words & Music , the museum offers an aggressive schedule of educational
programs . The museum also operates CMF Records , a Grammy - winning re-issue label ( The Complete Hank Williams and Night Train to Nashville : Music City
Rhythm & Blues , 1945 - 1970 ) ; and CMF Press , a book publishing arm that releases exhibit - related books in cooperation with Vanderbilt University Press and
other major trade publishing houses .
</item 1>

<item 2>
Country Music Hall of Fame and Museum
Located at the head of Music Row , the museum was erected on the site of a small Nashville city park . At this point , artifacts began to be displayed and a
small library was begun in a loft above one of the museum 's galleries . Early in the 1970s the basement of the museum building was partially complete , and
library expansion began , embracing not only recordings but also books and periodicals , sheet music and songbooks , photographs , business documents , and
other materials . At the outset , CMA staff had run the museum , but by 1972 the museum ( already governed by its own independent board of directors ) acquired
its own small staff , which has steadily increased to over 150 full - time professionals . Building expansion took place in 1974 , 1977 , and 1984 to store and
display the museum 's growing collection of costumes , films , historic cars , musical instruments , and other artifacts . An education department was created
to conduct ongoing programs with Middle Tennessee schools , an oral history program was begun , and a publications department was launched to handle books , as
well as the Journal of Country Music .
</item 2>

<item 3>
Country Music Hall of Fame and Museum
Contents ( hide ) 1 History of the museum 2 Current museum 3 The Country Music Hall of Fame 4 The Museum collection 5 Architecture and design 6 Timeline 7
Senior leadership 8 See also 9 Further reading 10 References 11 External links History of the museum ( edit ) The Country Music Hall of Fame and Museum is one
of the world 's largest and most active popular music research centers and the world 's largest repository of country music artifacts . Early in the 1960s , as
the Country Music Association 's campaign to publicize country music was accelerating , CMA leaders determined that a new organization was needed to operate a
country music museum and to carry out research and education activities beyond CMA 's scope as a trade organization . Toward this end , the nonprofit Country
Music Foundation ( CMF ) was chartered by the state of Tennessee in 1964 to collect , preserve , and publicize information and artifacts relating to the history
of country music . Through CMF , industry leaders raised money to build the Country Music Hall of Fame and Museum , which opened on April 1 , 1967 .
</item 3>

<item 4>
Country Music Hall of Fame and Museum
The Country Music Hall of Fame ( edit ) Membership in the Country Music Hall of Fame , the highest honor a country music professional can receive , is extended
to performers , songwriters , broadcasters , musicians , and executives in recognition of their contributions to the development of country music . The Country
Music Hall of Fame honor was created in 1961 by the Country Music Association ( CMA ) ; the first inductees were Hank Williams , Jimmie Rodgers , and Fred Rose
. Roy Acuff , the first living artist to join the Hall of Fame , was elected in 1962 . The most recent inductees ( class of 2017 ) are Jerry Reed , Don Schlitz
and Alan Jackson . Over the Hall of Fame 's history , the number of new members inducted each year has varied from one to twelve ( no nominee was inducted in
1963 , no candidate having received sufficient votes ) . The election procedure is as follows : A small CMA nominating committee drafts slates of candidates
from each category ; categories have been defined variously over the years .
</item 4>

<item 5>
Country Music Hall of Fame and Museum
Nashville , Tenn. : Country Music Foundation , 1990 . N.B. : The structure as described in this document is the original one which had opened in 1967 ; the book
( amply ill . in b&w and col . ) describes the work , collections , etc. of the center 's Hall of Fame , Museum ( which includes a library ) , and recording
studio , and also gives an account of the development and various aspects of the country music industry . ISBN 0 - 915608 - 12 - X References ( edit ) ^ Jump up
to : `` About '' . Country Music Hall of Fame . Retrieved January 14 , 2016 . Jump up ^ McCall , Michael ( 2012 ) . The Encyclopedia of Country Music Second
Edition . Oxford University Press . pp. 105 -- 106 . ISBN 0195395638 . Jump up ^ `` Home '' . Studio B. Retrieved August 26 , 2017 . Jump up ^ `` Country Music
Hall of Fame promotes four to VP roles '' . Nashville Post .
</item 5>

<item 6>
Country Music Hall of Fame and Museum
Timeline ( edit ) 1961 : Country Music Association established the Country Music Hall of Fame 1964 : CMA chartered the not - for - profit Country Music
Foundation ( CMF ) , which operates the Country Music Hall of Fame and Museum 1967 : Country Music Hall of Fame and Museum opened on Music Row 1987 : Museum
earned accreditation from the American Alliance of Museums 2000 : December 31 , museum closed its doors on Music Row 2001 : May 17 , museum opened in downtown
Nashville 2013 : October 12 , Hatch Show Print opened to the public 2014 : New expansion opened , more than doubling the size of the building 2015 : For the
first time , the museum had more than one million visitors in a calendar year Senior leadership ( edit ) Kyle Young , CEO Sharon Brawner , Senior Vice President
of Marketing and Sales Carolyn Tate , Senior Vice President of Museum Services Nina Burghard , Senior Vice President of Financial Services and Operations Lisa
Purcell , Vice President of Development See also ( edit ) Country music portal List of Country Music Hall of Fame inductees Further reading ( edit ) The Country
Music Hall of Fame & Museum .
</item 6>

<item 7>
Country Music Hall of Fame and Museum
Award recipients are determined through a two - stage balloting process ; the first round of voting narrows each category to five candidates ; the second round
selects winners . The large select committee of electors that votes on Hall of Fame membership is composed of CMA members who have participated in the country
music industry for at least ten years . New Hall of Fame members receive special recognition in ceremonies at the Country Music Hall of Fame and Museum . Bas -
relief portraits cast in bronze honoring each Hall of Fame member were originally displayed at the Tennessee State Museum in downtown Nashville until the
Country Music Hall of Fame and Museum opened its own building in April 1967 ; in this barn - roofed facility at the head of Music Row , the bronze plaques
comprised a special exhibit . Today the plaques are displayed in a seventy - foot - high rotunda at the museum 's enlarged downtown Nashville facility . The
museum collection ( edit ) The museum 's archival and library storage , allows for the cultivation of the museum 's collection and includes more than 2.5
million artifacts .
</item 7>

<item 8>
Country Music Hall of Fame and Museum
Country Music Hall of Fame and museum - wikipedia Country Music Hall of Fame and museum Jump to : navigation , search This article needs additional citations
for verification . Please help improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( May 2016 ) (
Learn how and when to remove this template message ) Country Music Hall of Fame and Museum Established 1961 ( 1961 ) Location 222 5th Ave S Nashville ,
Tennessee 37203 Type Hall of fame Visitors 1.1 million in 2015 Director Kyle Young Website www.countrymusichalloffame.org The Country Music Hall of Fame and
Museum in Nashville , Tennessee , is one of the world 's largest museums and research centers dedicated to the preservation and interpretation of American
vernacular music . Chartered in 1964 , the museum has amassed one of the world 's most extensive musical collections .
</item 8>

<item 9>
Country Music Hall of Fame and Museum
Current museum ( edit ) To become more accessible , in May 2001 the Country Music Hall of Fame and Museum moved to a new , 130,000 - square - foot ( 12,000 m )
facility in the heart of downtown Nashville 's arts and entertainment district . In 2014 , the museum unveiled a $100 million expansion , doubling its size to
350,000 square feet of galleries , archival storage , education classrooms , retail stores , and special event space . In the museum 's core exhibition , Sing
Me Back Home : A Journey Through Country Music , visitors are immersed in the history and sounds of country music , its origins and traditions , and the stories
and voices of many of its architects . The story is revealed through artifacts , photographs , and text panels , recorded sound , vintage video , and
interactive touchscreens . Sing Me Back Home is enhanced by numerous , rotating limited - engagement exhibits . The new ACM Gallery and the Dinah and Fred
Gretsch Family Gallery offer visitors a hands - on immersion into today 's country music with artifacts from today 's country stars and a series of technology -
enhanced activities .
</item 9>

<item 10>
Country Music Hall of Fame and Museum
Nashville Post . February 17 , 2015 . Retrieved January 19 , 2016 . External links ( edit ) Wikimedia Commons has media related to Country Music Hall of Fame
and Museum . Official website ( hide ) Nashville landmarks Existing AT&T Building Belmont Mansion Bicentennial Capitol Mall State Park Bluebird Café Bridgestone
Arena Centennial Park Cheekwood Country Music Hall of Fame First Tennessee Park Fort Nashborough Fort Negley Frist Center for the Visual Arts Gaylord Opryland
Resort Grand Ole Opry Greer Stadium The Hermitage Life & Casualty Tower Memorial Gymnasium Music City Center Nashville City Cemetery Nashville International
Airport Nashville Zoo at Grassmere Nissan Stadium Parthenon Ryman Auditorium Schermerhorn Symphony Center John Seigenthaler Pedestrian Bridge Tennessee
Performing Arts Center Tennessee State Capitol Tennessee State Museum Tennessee Tower Union Station Vanderbilt Stadium Former Opryland USA Starwood Amphitheatre
Sulphur Dell Coordinates : 36 ° 09 ′ 29.80 '' N 86 ° 46 ′ 34.18 '' W ﻿ / ﻿ 36.1582778 ° N 86.7761611 ° W ﻿ / 36.
</item 10>
=== END ITEM id=2e73e2a88d938bf5 ===

=== ITEM id=cc104e385cd00ead ===
Gold sentences and facts:
Sentence 0: Originally an ordinary man by the name of Nick Chopper ( the name first appearing in The Marvelous Land of Oz ) , the Tin Woodman used to make his living chopping down trees in the forests of Oz , as his father had before him . The Wicked Witch of the East enchanted his axe to prevent him from marrying his sweetheart , after being bribed by the lazy old woman who kept the Munchkin maiden as a servant , and did not wish to lose her . ( In a later book of the series , The Tin Woodman of Oz , the woman is said to be the Witch 's servant , and it is the Witch herself who decides to enchant Nick 's axe . ) The enchanted axe chopped off his limbs , one by one . Each time he lost a limb , Ku - Klip the tinsmith replaced it with a prosthetic limb made of tin . Finally , nothing was left of him but tin . However , Ku - Klip neglected to replace his heart . Once Nick Chopper was made entirely of tin , he was no longer able to love the lady he had fallen for .
  Fact 0.0: Nick Chopper was originally an ordinary man.
  Fact 0.1: The name Nick Chopper first appears in The Marvelous Land of Oz.
  Fact 0.2: Nick Chopper used to make his living chopping down trees in the forests of Oz.
  Fact 0.3: Nick Chopper's father had also chopped down trees in the forests of Oz.
  Fact 0.4: The Wicked Witch of the East enchanted Nick Chopper's axe to prevent him from marrying his sweetheart.
  Fact 0.5: The Wicked Witch of the East was bribed to do this by the lazy old woman who kept the Munchkin maiden as a servant.
  Fact 0.6: The lazy old woman did not wish to lose the Munchkin maiden servant.
  Fact 0.7: In a later book of the series, The Tin Woodman of Oz, the woman is said to be the Witch's servant.
  Fact 0.8: In The Tin Woodman of Oz, the Witch herself decides to enchant Nick's axe.
  Fact 0.9: The enchanted axe chopped off Nick Chopper's limbs one by one.
  Fact 0.10: Each time Nick lost a limb, Ku-Klip the tinsmith replaced it with a prosthetic limb made of tin.
  Fact 0.11: Finally, nothing was left of Nick Chopper but tin.
  Fact 0.12: Ku-Klip neglected to replace Nick Chopper's heart.
  Fact 0.13: Once Nick Chopper was made entirely of tin, he was no longer able to love the lady he had fallen for.
Sentence 1: In The Wonderful Wizard of Oz , Dorothy Gale befriends the Tin Woodman after they find him rusted in the forest , as he was caught in rain , and use his oil can to release him . He follows her to the Emerald City to get a heart from The Wizard . They are joined on their adventure by the Scarecrow and the Cowardly Lion . The Wizard sends Dorothy and her friends to the Winkie Country to kill the Wicked Witch of the West . The Tin Woodman 's axe proves useful in this journey , both for chopping wood to create a bridge or raft as needed , and for chopping the heads off animals that threaten the party . When the Winged monkeys are sent by the Witch of the West against the group , they throw the Tin Woodman from a great height , damaging him badly . However Winkie Tinsmiths are able to repair him after the death of the Witch .
  Fact 1.0: In The Wonderful Wizard of Oz, Dorothy Gale befriends the Tin Woodman.
  Fact 1.1: Dorothy and her friends find the Tin Woodman rusted in the forest.
  Fact 1.2: The Tin Woodman was caught in rain.
  Fact 1.3: Dorothy uses the Tin Woodman's oil can to release him.
  Fact 1.4: The Tin Woodman follows Dorothy to the Emerald City to get a heart from the Wizard.
  Fact 1.5: The Scarecrow and the Cowardly Lion join Dorothy and the Tin Woodman on their adventure.
  Fact 1.6: The Wizard sends Dorothy and her friends to the Winkie Country to kill the Wicked Witch of the West.
  Fact 1.7: The Tin Woodman's axe proves useful on this journey for chopping wood to create a bridge or raft as needed.
  Fact 1.8: The Tin Woodman's axe is useful for chopping the heads off animals that threaten the party.
  Fact 1.9: The Winged Monkeys are sent by the Witch of the West against the group.
  Fact 1.10: The Winged Monkeys throw the Tin Woodman from a great height.
  Fact 1.11: The fall damages the Tin Woodman badly.
  Fact 1.12: Winkie Tinsmiths are able to repair the Tin Woodman after the death of the Witch.
Sentence 2: His desire for a heart notably contrasts with the Scarecrow 's desire for brains , reflecting a common debate between the relative importance of the mind and the emotions . This occasions philosophical debate between the two friends as to why their own choices are superior ; neither convinces the other , and Dorothy , listening , is unable to decide which one is right . Symbolically , because they remain with Dorothy throughout her quest , she is provided with both and need not select . The Tin Woodman states unequivocally that he has neither heart nor brain , but cares nothing for the loss of his brain . Towards the end of the novel , though , Glinda praises his brain as not quite that of the Scarecrow 's .
  Fact 2.0: The Tin Woodman's desire for a heart notably contrasts with the Scarecrow's desire for brains.
  Fact 2.1: The contrast reflects a common debate between the relative importance of the mind and the emotions.
  Fact 2.2: The two friends have a philosophical debate about why their own choices are superior.
  Fact 2.3: Neither friend convinces the other in that debate.
  Fact 2.4: Dorothy, listening, is unable to decide which one is right.
  Fact 2.5: Symbolically, because the Tin Woodman and the Scarecrow remain with Dorothy throughout her quest, she is provided with both a heart and a brain and need not select.
  Fact 2.6: The Tin Woodman states unequivocally that he has neither heart nor brain.
  Fact 2.7: The Tin Woodman cares nothing for the loss of his brain.
  Fact 2.8: Towards the end of the novel, Glinda praises the Tin Woodman's brain.
  Fact 2.9: Glinda says the Tin Woodman's brain is not quite that of the Scarecrow.

Retrieved items:
<item 1>
Tin Woodman
) The Magical Mimics in Oz ( 1946 ) The Shaggy Man of Oz ( 1949 ) The Hidden Valley of Oz ( 1951 ) Merry Go Round in Oz ( 1963 ) Yankee in Oz ( 1972 ) The
Enchanted Island of Oz ( 1976 ) The Forbidden Fountain of Oz ( 1980 ) The Ozmapolitan of Oz ( 1986 ) The Wicked Witch of Oz ( 1993 ) The Giant Garden of Oz (
1993 ) The Runaway in Oz ( 1995 ) The Rundelstone of Oz ( 2000 ) The Emerald Wand of Oz ( 2005 ) Trouble Under Oz ( 2006 ) Alternate Dorothy of Oz ( 1989 )
Locations and elements Land of Oz Emerald City Munchkin Country Gillikin Country Winkie Country Quadling Country Yellow brick road Oogaboo Deadly Desert
Nonestica Forest of Burzee Land of Ev Merryland Mo Ix Nome Kingdom Noland Ruby slippers Silver Shoes The Oz Film Manufacturing Company Characters Dorothy Gale
Toto Princess Ozma The Wizard of Oz Scarecrow Tin Woodman Cowardly Lion Jack Pumpkinhead Tik - Tok Sawhorse Trot Hungry Tiger
</item 1>

<item 2>
Tin Woodman
The movie shows him moving through a colorful Oz city with his brand - new mechanical heart , before meeting a reddish female Tin Woman ( or robot ? ) to whom
he gives the heart . She proceeds to cruelly play with the heart . An internet - collaboration , CG animated feature based on Baum 's book The Tin Woodman of Oz
was produced by A : M Films , and completed in 2009 . In 2010 , Whitestone Motion Pictures produced a 23 - minute live - action short film , Heartless : The
Story of the Tin Man . The film is based on the book The Tin Woodman of Oz , where Woot the Wanderer visits the Tin Man and asks how he came to be made of tin .
In the film , the Tin Man appears to be more steam - or coal - powered . His chest cavity is covered by a door which reads `` Pedudoe Tin Co . '' but this is a
reference to the workings of the film company , and not a reference to any Oz book or material . The movie was made available for free viewing online and free
downloading of its soundtrack .
</item 2>

<item 3>
The Tin Woodman uses his axe to build crossings and defend the party. The Winged Monkeys badly damage him, but Winkie tinsmiths repair him after the Witch’s
death. He longs for a heart, contrasting with the Scarecrow’s desire for brains; their debate about emotion versus intellect remains unresolved. Symbolically,
Dorothy’s companions give her both. Though the Woodman says he has neither heart nor brain and does not miss his brain, Glinda later praises his
</item 3>

<item 4>
Tin Woodman
The Wizard turns out to be a `` humbug '' and can only provide a placebo heart made of silk and filled with sawdust . However , this is enough to please the Tin
Woodman , who , with or without a heart , was all along the most tender and emotional of Dorothy 's companions ( just as the Scarecrow was the wisest and the
Cowardly Lion the bravest ) . When he accidentally crushes an insect , he is grief - stricken and , ironically , claims that he must be careful about such
things , while those with hearts do not need such care . This tenderness remains with him throughout the series , as in The Patchwork Girl of Oz , where he
refuses to let a butterfly be maimed for the casting of a spell . When Dorothy returns home to her farm in Kansas , the Tin Woodman returns to the Winkie
Country to rule as emperor . Later , he has his subjects construct a palace made entirely of tin -- from the architecture all the way down to the flowers in the
garden . Baum emphasized that the Tin Woodman remains alive , in contrast to the windup mechanical man Tik - Tok Dorothy meets in a later book .
</item 4>

<item 5>
Tin Woodman
( In a later book of the series , The Tin Woodman of Oz , the woman is said to be the Witch 's servant , and it is the Witch herself who decides to enchant Nick
's axe . ) The enchanted axe chopped off his limbs , one by one . Each time he lost a limb , Ku - Klip the tinsmith replaced it with a prosthetic limb made of
tin . Finally , nothing was left of him but tin . However , Ku - Klip neglected to replace his heart . Once Nick Chopper was made entirely of tin , he was no
longer able to love the lady he had fallen for . In The Wonderful Wizard of Oz , Dorothy Gale befriends the Tin Woodman after they find him rusted in the forest
, as he was caught in rain , and use his oil can to release him . He follows her to the Emerald City to get a heart from The Wizard . They are joined on their
adventure by the Scarecrow and the Cowardly Lion . The Wizard sends Dorothy and her friends to the Winkie Country to kill the Wicked Witch of the West .
</item 5>

<item 6>
Stanum seeks Zelena’s help in Oz to prevent becoming tin. They find the Crimson Heart, but it requires absorbing another person’s magic. Suspicious that Stanum
is working with Dorothy, Zelena takes the heart for herself, leaving him to become permanently tin; she later leaves him immobilized, and he is eventually seen
torn apart beside the Yellow Brick Road. The song “Country Robot / A Letter to Dorothy” by The Incredible Moses Leroy is written from the Tin Man’s
</item 6>

<item 7>
Tin Woodman
When they get to the heart , Zelena picks it up and felt some of her magic draining . When Zelena claims that Dorothy Gale put Stanum up to this task , Stanum
commented that he never met Dorothy . He also stated that Zelena once claimed on the day they met that she would be happy being a normal human . As Stanum fully
turns to tin and is unable to move , Zelena takes the box containing the Crystal Heart and leaves while telling Stanum that perhaps Dorothy will meet Stanum and
free him . Modern works ( edit ) This article gives self - sourcing examples without describing their significance in the context of the article . Please help
improve this article by adding citations to reliable sources that describe the examples ' significance , and by removing less pertinent examples . Unsourced or
poorly sourced material may be challenged or removed . ( November 2017 ) ( Learn how and when to remove this template message ) The song `` Country Robot / A
Letter to Dorothy '' by The Incredible Moses Leroy is written from the Tin Man 's perspective ; it includes the lyrics `` You gave me oil , I was a rusty load /
You even helped me find my heart . ''
</item 7>

<item 8>
Tin Woodman
They built a story around them , added Dorothy , and added a series of lessons to the effect that everyone possesses the resources they need ( such as brains ,
a heart and courage ) if only they had self - confidence . The Wonderful Wizard of Oz was a children 's book , of course , but as Baum warned in the preface ,
it was a `` modernized '' fairy tale as well . The Tin Man -- the human turned into a machine -- was a common feature in political cartoons and in
advertisements in the 1890s . Indeed , he had been part of European folk art for 300 years . In political interpretations of The Wonderful Wizard of Oz , the
Tin Woodman is supposedly described as a worker , dehumanized by industrialization . The Tin Woodman little by little lost his natural body and had it replaced
by metal ; so he has lost his heart and can not move without the help of farmers ( represented by the Scarecrow ) ; in reality he has a strong sense of
cooperation and love , which needs only an infusion of self - confidence to be awakened . In the 1890s many argued that to secure a political revolution a
coalition of Farmers and Workers was needed .
</item 8>

<item 9>
Tin Woodman
To prevent himself from completely becoming tin , Stanum seeks out help from Zelena ( now the Wicked Witch of the West ) at the Emerald City of Oz . Zelena
agrees to help him seek out the Crimson Heart , which can save him . During their quest , Stanum tells Zelena that she does n't have to be lonely but she is
doing her best to deny his advice . Suddenly , a lion comes of nowhere to attack Stanum , and Zelena uses her magic to make the lion go away ( at this point the
lion has become as Zelena would put it , cowardly ) . When they finally arrived to the location of the Crimson Heart , the two learned that the only way to make
it work is through the absorption of another person 's magic . Unfortunately Zelena 's actions and selfish greed for magic causes her to betray Stanum , whom
she suspect was aligned with Dorothy by keeping the Crimson Heart for herself , leaving Stanum to transform into the Tin Man permanently . Some time later ,
when Robin Hood arrives in Oz in the episode `` Heart of Gold '' , the Tin Man is seen on the side of the Yellow Brick Road , torn apart .
</item 9>

<item 10>
Nick Chopper’s enchanted axe gradually severed his limbs, which tinsmith Ku-Klip replaced with tin; forgetting his heart, he left Nick unable to love. Dorothy
and her companions find him rusted, oil him, and take him to the Wizard seeking a heart. Though the Wizard is a humbug and gives him a sawdust-filled silk
heart, the Tin Woodman is already exceptionally tender and compassionate. After Dorothy returns to Kansas, he rules the Winkies and builds a palace
</item 10>
=== END ITEM id=cc104e385cd00ead ===

=== ITEM id=d9cf4670deae9636 ===
Gold sentences and facts:
Sentence 0: Beijing ( / beɪˈdʒɪŋ / ; Mandarin : ( pèi. tɕíŋ ) ( listen ) ) , formerly romanized as Peking , is the capital of the People 's Republic of China , the world 's second most populous city proper , and most populous capital city . The city , located in northern China , is governed as a direct - controlled municipality under the national government with 16 urban , suburban , and rural districts . Beijing Municipality is surrounded by Hebei Province with the exception of neighboring Tianjin Municipality to the southeast ; together the three divisions form the Jingjinji metropolitan region and the national capital region of China .
  Fact 0.0: Beijing is the capital of the People's Republic of China.
  Fact 0.1: Beijing was formerly romanized as Peking.
  Fact 0.2: Beijing is the world's second most populous city proper.
  Fact 0.3: Beijing is the most populous capital city.
  Fact 0.4: Beijing is located in northern China.
  Fact 0.5: Beijing is governed as a direct-controlled municipality under the national government.
  Fact 0.6: Beijing has 16 urban, suburban, and rural districts.
  Fact 0.7: Beijing Municipality is surrounded by Hebei Province, except for the neighboring Tianjin Municipality to the southeast.
  Fact 0.8: Beijing Municipality, Hebei Province, and Tianjin Municipality together form the Jingjinji metropolitan region.
  Fact 0.9: Beijing Municipality, Hebei Province, and Tianjin Municipality together form the national capital region of China.
Sentence 1: Over the past 3,000 years , the city of Beijing has had numerous other names . The name Beijing , which means `` Northern Capital '' ( from the Chinese characters 北 for north and 京 for capital ) , was applied to the city in 1403 during the Ming dynasty to distinguish the city from Nanjing ( the `` Southern Capital '' ) . The English spelling is based on the pinyin romanization of the two characters as they are pronounced in Standard Mandarin . An older English spelling , Peking , is the postal romanization of the same two characters as they are pronounced in Chinese dialects spoken in the southern port towns first visited by European traders and missionaries . Those dialects preserve the Middle Chinese pronunciation of 京 as kjaeng , prior to a phonetic shift in the northern dialects to the modern pronunciation . Although Peking is no longer the common name for the city , some of the city 's older locations and facilities , such as Beijing Capital International Airport , with IATA Code PEK , and Peking University , still use the former romanization .
  Fact 1.0: Over the past 3,000 years, the city of Beijing has had numerous other names.
  Fact 1.1: The name Beijing means 'Northern Capital'.
  Fact 1.2: The name Beijing is made of the Chinese characters 北 (north) and 京 (capital).
  Fact 1.3: The name Beijing was applied to the city in 1403 during the Ming dynasty.
  Fact 1.4: The name Beijing was applied to distinguish the city from Nanjing, the 'Southern Capital'.
  Fact 1.5: The English spelling Beijing is based on the pinyin romanization of the two characters as pronounced in Standard Mandarin.
  Fact 1.6: The older English spelling Peking is the postal romanization of the same two characters as pronounced in Chinese dialects spoken in the southern port towns.
  Fact 1.7: The southern port towns were the first visited by European traders and missionaries.
  Fact 1.8: The Chinese dialects spoken in the southern port towns preserve the Middle Chinese pronunciation of 京 as kjaeng.
  Fact 1.9: That pronunciation of 京 as kjaeng comes before a phonetic shift in the northern dialects to the modern pronunciation.
  Fact 1.10: Peking is no longer the common name for the city.
  Fact 1.11: Some of the city's older locations and facilities still use the former romanization Peking.
  Fact 1.12: Beijing Capital International Airport, with IATA code PEK, still uses the former romanization.
  Fact 1.13: Peking University still uses the former romanization.

Retrieved items:
<item 1>
Beijing
Towns within Beijing Municipality but outside the urban area include ( but are not limited to ) : Changping 昌平 Huairou 怀柔 Miyun 密云 Liangxiang 良 乡 Liulimiao 琉璃 庙
Tongzhou 通州 Yizhuang 亦庄 Tiantongyuan 天通苑 Beiyuan 北 苑 Xiaotangshan 小 汤山 Several place names in Beijing end with mén ( 门 ) , meaning `` gate '' , as they were the
locations of gates in the former Beijing city wall . Other place names end in cūn ( 村 ) , meaning `` village '' , as they were originally villages outside the
city wall . Neighborhoods This section contains embedded lists that may be poorly defined , unverified or indiscriminate . Please help to clean it up to meet
Wikipedia 's quality standards . Where appropriate , incorporate items into the main body of the article . ( October 2017 ) The Niujie Mosque is an important
historical attraction Main article : Neighborhoods in Beijing Neighborhoods may extend across multiple districts . Major neighborhoods in urban Beijing include
:
</item 1>

<item 2>
Beijing
And under the Hongwu Emperor 's feudal policies it was given to Zhu Di , one of his sons , who was created `` Prince of Yan '' . The early death of Zhu
Yuanzhang 's heir led to a succession struggle on his death , one that ended with the victory of Zhu Di and the declaration of the new Yongle era . Since his
harsh treatment of the Ming capital Yingtian ( modern Nanjing ) alienated many there , he established his fief as a new co-capital . The city of Beiping became
Beijing ( northern capital ) or Shuntian in 1403 . The construction of the new imperial residence , the Forbidden City , took from 1406 to 1420 ; this period
was also responsible for several other of the modern city 's major attractions , such as the Temple of Heaven and Tian'anmen . On 28 October 1420 , the city was
officially designated the capital of the Ming dynasty in the same year that the Forbidden City was completed . Beijing became the empire 's primary capital and
Yingtian , also called Nanjing ( southern capital ) , became the co-capital .
</item 2>

<item 3>
Beijing
tɕíŋ ) ( listen ) ) , formerly romanized as Peking , is the capital of the People 's Republic of China , the world 's second most populous city proper , and
most populous capital city . The city , located in northern China , is governed as a direct - controlled municipality under the national government with 16
urban , suburban , and rural districts . Beijing Municipality is surrounded by Hebei Province with the exception of neighboring Tianjin Municipality to the
southeast ; together the three divisions form the Jingjinji metropolitan region and the national capital region of China . As a city combining both modern and
traditional architecture , Beijing is a megacity rich in history , exemplified in its global influence in politics , economy , education , history , culture ,
and technology . Beijing is the second largest Chinese city by urban population after Shanghai and is the nation 's political , cultural , and educational
center . It is home to the headquarters of most of China 's largest state - owned companies and is a major hub for the national highway , expressway , railway ,
and high - speed rail networks .
</item 3>

<item 4>
Beijing
Retrieved 22 July 2009 . External links Find more aboutBeijingat Wikipedia 's sister projects Definitions from Wiktionary Media from Wikimedia Commons News from
Wikinews Quotations from Wikiquote Texts from Wikisource Textbooks from Wikibooks Travel guide from Wikivoyage Learning resources from Wikiversity Economic
profile for Beijing at HKTDC Visit Beijing Facebook Page Photograph of The approach to Peking -- outside the walls taken in 1890 by Sir Henry Norman Preceded by
Lin'an ( Song dynasty ) Capital of China ( as Dadu of Yuan ) 1264 -- 1368 Succeeded by Nanjing ( Ming dynasty ) Preceded by Nanjing ( Ming dynasty ) Capital of
China 1420 -- 1928 Succeeded by Nanjing ( ROC ) Preceded by Nanjing ( ROC ) Capital of the People 's Republic of China 1949 -- present Succeeded by present
capital Places adjacent to Beijing Zhangjiakou , Hebei Chengde , Hebei Chengde , Hebei Zhangjiakou , Hebei Beijing Langfang , Hebei Baoding , Hebei Langfang ,
</item 4>

<item 5>
Beijing
The name Beijing , which means `` Northern Capital '' ( from the Chinese characters 北 for north and 京 for capital ) , was applied to the city in 1403 during the
Ming dynasty to distinguish the city from Nanjing ( the `` Southern Capital '' ) . The English spelling is based on the pinyin romanization of the two
characters as they are pronounced in Standard Mandarin . An older English spelling , Peking , is the postal romanization of the same two characters as they are
pronounced in Chinese dialects spoken in the southern port towns first visited by European traders and missionaries . Those dialects preserve the Middle Chinese
pronunciation of 京 as kjaeng , prior to a phonetic shift in the northern dialects to the modern pronunciation . Although Peking is no longer the common name for
the city , some of the city 's older locations and facilities , such as Beijing Capital International Airport , with IATA Code PEK , and Peking University ,
still use the former romanization . The single Chinese character abbreviation for Beijing is 京 , which appears on automobile license plates in the city . The
official Latin alphabet abbreviation for Beijing is `` BJ '' .
</item 5>

<item 6>
Beijing
The official Latin alphabet abbreviation for Beijing is `` BJ '' . History Main article : History of Beijing Early History The earliest traces of human
habitation in the Beijing municipality were found in the caves of Dragon Bone Hill near the village of Zhoukoudian in Fangshan District , where Peking Man lived
. Homo erectus fossils from the caves date to 230,000 to 250,000 years ago . Paleolithic Homo sapiens also lived there more recently , about 27,000 years ago .
Archaeologists have found neolithic settlements throughout the municipality , including in Wangfujing , located in downtown Beijing . The first walled city in
Beijing was Jicheng , the capital city of the state of Ji and was built in 1045 BC . Within modern Beijing , Jicheng was located around the present Guang'anmen
area in the south of Xicheng District . This settlement was later conquered by the state of Yan and made its capital . Early Imperial China The Tianning Pagoda
, built around 1120 during the Liao dynasty . After the First Emperor unified China , Jicheng became a prefectural capital for the region .
</item 6>

<item 7>
Beijing
Hebei Beijing Langfang , Hebei Baoding , Hebei Langfang , Hebei Tianjin Articles related to Beijing ( hide ) Beijing History Politics Geography Toponym
Administration Districts Dongcheng Xicheng Chaoyang Haidian Fengtai Shijingshan Mentougou Fangshan Tongzhou Shunyi Changping Daxing Huairou Pinggu Miyun Yanqing
Defunct Chongwen Xuanwu Yanshan Other Administrative divisions Township - level divisions Economic and technological development zones Education List of
universities Peking University Tsinghua University Beijing Normal University Renmin University Transportation Ring roads Expressways Highways Subway Suburban
Railway Bus Airports : Capital International , Nanyuan , Daxing ( under construction ) Railway stations : Beijing West , Beijing , Beijing East , Beijing South
, Beijing North , Shahe ,
</item 7>

<item 8>
Beijing
Beijing - wikipedia Beijing Jump to : navigation , search `` Peking '' redirects here . For other uses , see Beijing ( disambiguation ) and Peking (
disambiguation ) . Beijing 北京 市 Municipality Beijing Municipality Clockwise from top : Beijing CBD skyline , Tiananmen , Temple of Heaven , National Center for
the Performing Arts , and Beijing National Stadium Location of Beijing Municipality within China Coordinates : 39 ° 55 ′ N 116 ° 23 ′ E ﻿ / ﻿ 39.917 ° N 116.383
° E ﻿ / 39.917 ; 116.383 Coordinates : 39 ° 55 ′ N 116 ° 23 ′ E ﻿ / ﻿ 39.917 ° N 116.383 ° E ﻿ / 39.917 ; 116.383 Country China Established date 1045 BCE ( Zhou
Dynasty ) Divisions - County - level - Township - level 16 districts 289 towns and villages Government Type Municipality Party Secretary Cai Qi Mayor Chen
Jining Congress Chairman Li Wei ( zh ) Conference Chairman Ji Lin Area Municipality 16,411 km ( 6,336 sq mi ) Land 16,
</item 8>

<item 9>
Beijing
1 Towns 4.1. 2 Neighborhoods 4.2 Judiciary and procuracy 4.3 Diplomatic missions 5 Economy 5.1 Sector composition 5.2 Economic zones 6 Demographics 6.1
Metropolitan area 7 Culture 7.1 Places of interest 8 Religion 8.1 Chinese folk religion and Taoism 8.2 Buddhism 8.3 Islam 8.4 Christianity 9 Media 9.1
Television and radio 9.2 Press 10 Sports 10.1 Events 10.2 Venues 10.3 Clubs 11 Transportation 11.1 Rail and high - speed rail 11.2 Roads and expressways 11.3
Air 11.4 Public transit 11.5 Taxi 11.6 Bicycles 12 Defense and aerospace 13 Nature and wildlife 14 See also 15 References 16 Further reading 17 External links
Etymology See also : Names of Beijing Over the past 3,000 years , the city of Beijing has had numerous other names .
</item 9>

<item 10>
Beijing
1928 -- 1937 ; 1945 -- 1949 ) Literal meaning `` Northern Capital '' ( show ) Transcriptions Standard Mandarin Hanyu Pinyin Běijīng Bopomofo ㄅ ㄟ ˇ ㄐ ㄧ ㄥ Gwoyeu
Romatzyh Beeijing Wade -- Giles Pei - ching IPA ( pèi. tɕíŋ ) Wu Suzhounese Poh - cin Hakka Romanization Bet - gin Yue : Cantonese Yale Romanization Bākgìng or
Bākgīng IPA ( pɐ́k̚. kêŋ ) or ( pɐ́k̚. kéŋ ) Jyutping Bak1ging1 Southern Min Hokkien POJ Pak - kian Tâi - lô Pak - kiann Eastern Min Fuzhou BUC Báe̤k - gĭng
Beijing ( / beɪˈdʒɪŋ / ; Mandarin : ( pèi.
</item 10>
=== END ITEM id=d9cf4670deae9636 ===
