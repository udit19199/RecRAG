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

=== ITEM id=6c0878056b78823b ===
Gold sentences and facts:
Sentence 0: The states that have legislatively adopted stand - your - ground laws are Alabama , Alaska , Arizona , Florida , Georgia , Idaho , Indiana , Iowa , Kansas , Kentucky , Louisiana , Michigan , Mississippi , Missouri , Montana , Nevada , New Hampshire , North Carolina , Oklahoma , Pennsylvania , South Carolina , South Dakota , Tennessee , Texas , Utah , West Virginia , and Wyoming .
  Fact 0.0: Alabama has legislatively adopted stand-your-ground laws.
  Fact 0.1: Alaska has legislatively adopted stand-your-ground laws.
  Fact 0.2: Arizona has legislatively adopted stand-your-ground laws.
  Fact 0.3: Florida has legislatively adopted stand-your-ground laws.
  Fact 0.4: Georgia has legislatively adopted stand-your-ground laws.
  Fact 0.5: Idaho has legislatively adopted stand-your-ground laws.
  Fact 0.6: Indiana has legislatively adopted stand-your-ground laws.
  Fact 0.7: Iowa has legislatively adopted stand-your-ground laws.
  Fact 0.8: Kansas has legislatively adopted stand-your-ground laws.
  Fact 0.9: Kentucky has legislatively adopted stand-your-ground laws.
  Fact 0.10: Louisiana has legislatively adopted stand-your-ground laws.
  Fact 0.11: Michigan has legislatively adopted stand-your-ground laws.
  Fact 0.12: Mississippi has legislatively adopted stand-your-ground laws.
  Fact 0.13: Missouri has legislatively adopted stand-your-ground laws.
  Fact 0.14: Montana has legislatively adopted stand-your-ground laws.
  Fact 0.15: Nevada has legislatively adopted stand-your-ground laws.
  Fact 0.16: New Hampshire has legislatively adopted stand-your-ground laws.
  Fact 0.17: North Carolina has legislatively adopted stand-your-ground laws.
  Fact 0.18: Oklahoma has legislatively adopted stand-your-ground laws.
  Fact 0.19: Pennsylvania has legislatively adopted stand-your-ground laws.
  Fact 0.20: South Carolina has legislatively adopted stand-your-ground laws.
  Fact 0.21: South Dakota has legislatively adopted stand-your-ground laws.
  Fact 0.22: Tennessee has legislatively adopted stand-your-ground laws.
  Fact 0.23: Texas has legislatively adopted stand-your-ground laws.
  Fact 0.24: Utah has legislatively adopted stand-your-ground laws.
  Fact 0.25: West Virginia has legislatively adopted stand-your-ground laws.
  Fact 0.26: Wyoming has legislatively adopted stand-your-ground laws.

Retrieved items:
<item 1>
Stand-your-ground law
See also ( edit ) Jury nullification Proxemics , the study of human use of space and the effects that population density has on behavior , communication , and
social interaction Restorative justice References ( edit ) ^ Jump up to : Dawkins v. State , 252 P. 3d 214 ( Okla . 2011 ) Jump up ^ Randall , Mark ; DeBoer ,
Hendrick ( April 24 , 2012 ) . `` The Castle Doctrine and Stand - Your - Ground Law '' . Jump up ^ `` Duty to Retreat '' . Wikipedia . Jump up ^ Florida
Statutes Title XLVI Chapter 776 Jump up ^ Ala . Code 13A - 3 - 23 ( b ) : `` A person who is justified under subsection ( a ) in using physical force ,
including deadly physical force , and who is not engaged in an unlawful activity and is in any place where he or she has the right to be has no duty to retreat
and has the right to stand his ground . '' Jump up ^ Eaton , Daysha ( 2013 - 06 - 20 ) .
</item 1>

<item 2>
Stand-your-ground law
The castle doctrine is an exception to this . It gives immunity from liability to individuals who acted in self - defense in the home even if they could have
safely retreated from the threat and failed to do so . The duty to retreat is a legal requirement in some jurisdictions that a threatened person can not stand
one 's ground and apply lethal force in self - defense , but must retreat to a place of safety instead . Deadly force or lethal force is force with the intent
of serious bodily injury or death to another person . In most jurisdictions it is only accepted under conditions of extreme necessity and last resort . Stand -
your - ground laws eliminate the retreat requirement at any location the defendant has a legal right to be , though this varies from state to state .
</item 2>

<item 3>
Stand-your-ground law
Stand - your - ground law - wikipedia Stand - your - ground law Jump to : navigation , search `` Stand your ground '' redirects here . For other uses , see
Stand Your Ground ( disambiguation ) . A stand - your - ground law ( sometimes called `` line in the sand '' or `` no duty to retreat '' law ) is a
justification in a criminal case , whereby defendants can `` stand their ground '' and use force without retreating , in order to protect and defend themselves
or others against threats or perceived threats . An example is where there is no duty to retreat from any place where they have a lawful right to be , and that
they may use any level of force if they reasonably believe the threat rises to the level of being an imminent and immediate threat of serious bodily harm and or
death . One case describes `` the ' stand your ground ' law ... a person has a right to expect absolute safety in a place they have a right to be , and may use
deadly force to repel an unlawful intruder . ''
</item 3>

<item 4>
Stand-your-ground law
Although police officers make up for additional homicides , they are justified . These are not classified as murder because the officers are acting in the line
of duty . Police are not required to retreat when acting in the line of duty . A 2015 study found that the adoption of Oklahoma 's stand - your - ground law was
associated with a decrease in residential burglaries , but also that the law had `` the unintended consequence of increasing the number of non-residential
burglaries . '' In 2016 , Mark Gius published a study that found that stand - your - ground laws were not associated with lower crime rates . A widely reported
2016 study in the Journal of the American Medical Association compared homicide rates in Florida following the passage of its `` stand your ground '' self -
defense law to the rates in four control states , New Jersey , New York , Ohio and Virginia , which have no similar laws . It found that the law was associated
with a 24.4 % increase in homicide and a 31.6 % increase in firearm - related homicide , but no change in rates of suicide or suicide by firearm , between 2005
and 2014 .
</item 4>

<item 5>
Stand-your-ground law
At English common law there is no duty to retreat before a person may use reasonable force against an attacker , nor need a person wait to be attacked before
using such force , but one who chooses not to retreat , when retreat would be a safe and easy option , might find it harder to justify his use of force as '
reasonable ' . Any force used must be reasonable in the circumstances as the person honestly perceived them to be , after making allowance for the fact that
some degree of excess force might still be reasonable in the heat of the moment . It was once thought that in ' householder defence ' cases , excessive force
would be regarded as unreasonable only if `` grossly disproportionate '' , but that view has twice been rejected by the courts . Grossly disproportionate force
can never be reasonable , but lesser force might or might not be , depending on the circumstances . In R v Ray ( 2017 ) the Lord Chief Justice , giving the
judgment of the Court of Appeal , said : `` If there is a threat of confrontation in the street , then the option to retreat may be important in determining
whether the use of any force was reasonable .
</item 5>

<item 6>
Stand-your-ground law
States that have adopted stand - your - ground , but limit it to only when a person is within their vehicle , are North Dakota , Ohio , and Wisconsin . The
states that have castle doctrine only with the duty to retreat in public are Arkansas , Connecticut , Delaware , Maine , Maryland , Massachusetts , Minnesota ,
Nebraska , New York , New Jersey , and Rhode Island . This means that people can use deadly force in their home , car , or other form of abode but have to
retreat in public . Controversy ( edit ) Stand - your - ground laws are frequently criticized and called `` shoot first '' laws by critics , including the Brady
Campaign to Prevent Gun Violence . In Florida , self - defense claims tripled in the years following enactment . The law 's critics argue that Florida 's law
makes it very difficult to prosecute cases against individuals who shoot others and then claim self - defense . The shooter can argue that he felt threatened ,
and in most cases , the only witness who could have argued otherwise is the deceased .
</item 6>

<item 7>
Stand-your-ground law
Justification using stand - your - ground laws may be limited when `` ( the defendant ) was engaged in illegal activities and not entitled to benefit from
provisions of the ' stand your ground ' law '' . This may be the case even if the illegal conduct the defendant was engaged in had nothing to do with the threat
which instigated his use of deadly force . For instance , using an illegally obtained weapon in self defense to avoid being robbed and beaten is not justified
by this law . The castle doctrine is a common law doctrine stating that persons have no duty to retreat in their home , or `` castle '' , and may use reasonable
force , including deadly force , to defend their property , person , or another . Outside of the abode , however , a person has a duty to retreat , if possible
, before using deadly force . Castle doctrine and `` stand - your - ground '' laws are acceptable defenses for people who have been charged with criminal
homicide . At common law , self - defense claims are not valid if the defendant could have safely retreated from danger ( duty to retreat ) . The castle
doctrine is an exception to this .
</item 7>

<item 8>
Stand-your-ground law
Jump up ^ Coleman , Christina . `` Kill At Will : Stand Your Ground Laws Contribute To 600 Additional Homicides A Year ( DETAILS ) '' . Retrieved 1 December
2014 . Jump up ^ Roberts , Joshua ( Winter 2012 ) . `` Deadly Force and the Right of Self - Defense Stand Your Ground Laws '' . Forensic Examiner -- via
ProQuest . Jump up ^ Chamlin , Mitchell B. ; Krajewski , Andrea E. ( 29 December 2015 ) . `` Use of Force and Home Safety : An Impact Assessment of Oklahoma 's
'' . Deviant Behavior : 1 -- 9 . doi : 10.1080 / 01639625.2015. 1012027 . Jump up ^ Gius , Mark ( September 2016 ) . `` The relationship between stand - your -
ground laws and crime : A state - level analysis '' . The Social Science Journal . 53 ( 3 ) : 329 -- 338 . doi : 10.1016 / j. soscij. 2016.01. 001 .
</item 8>

<item 9>
Stand-your-ground law
Contents ( hide ) 1 United States 1.1 Laws 1.2 Controversy 1.3 Racial disparity 1.4 Crime 2 Other jurisdictions 2.1 England and Wales 2.2 Czech Republic 2.3
Germany 2.4 Ireland 3 See also 4 References 5 Further reading United states ( edit ) Laws ( edit ) Stand your ground law by US jurisdiction Stand - your -
ground law Stand - your - ground in practice Stand - your - ground from within one 's vehicle Castle doctrine only ; duty to retreat in public Duty to retreat
The states that have legislatively adopted stand - your - ground laws are Alabama , Alaska , Arizona , Florida , Georgia , Idaho , Indiana , Iowa , Kansas ,
Kentucky , Louisiana , Michigan , Mississippi , Missouri , Montana , Nevada , New Hampshire , North Carolina , Oklahoma , Pennsylvania , South Carolina , South
Dakota , Tennessee , Texas , Utah , West Virginia , and Wyoming . The states that have adopted stand - your - ground in practice , either through case law /
precedent , jury instructions or by other means , are California , Colorado , Illinois , New Mexico , Oregon , Virginia , and Washington .
</item 9>

<item 10>
Stand-your-ground law
Florida governor Rick Scott plans his own investigation into the law . In a July 16 , 2013 speech in the wake of the jury verdict acquitting George Zimmerman of
charges stemming from the shooting death of Trayvon Martin , Attorney General Eric Holder criticized stand - your - ground laws as `` senselessly expand ( ing )
the concept of self - defense and sow ( ing ) dangerous conflict in our neighborhoods . '' The defendant , George Zimmerman , claims he was restrained at the
time of the shooting , thus allowing no option for retreat and making ' stand your ground ' irrelevant to the case . Florida 's legislature is currently
considering a bill that would allow people to show a gun or fire a warning shot during a confrontation without drawing a lengthy prison sentence . There is a
bill moving through Republican legislature that would require the prosecution to prove that the defendant 's use of self - defense is not valid . If this passes
, prosecutors will have to try a case twice at a hearing and a trial which would make it easier for defendants to argue against prosecution on the basis of
their fears .
</item 10>
=== END ITEM id=6c0878056b78823b ===

=== ITEM id=81abcd530d79365a ===
Gold sentences and facts:
Sentence 0: Place Actor / Actress Episodes Finals Total score ( 175 ) 3 - 4 5 - 6 7 - 8 9 - 10 11 - 12 13 - 14 15 - 16 17 - 18 19 - 20 21 - 22 23 - 24 25 - 26 Aman Gandotra 8 12 14 14 7 + 3 = 10 9 + 3 = 12 10 + 3 = 13 10 + 4 = 14 10 + 5 = 15 10 + 5 = 15 10 + 4 = 14 Winner 141 Natasha Bharadwaj 9 18 14 15 10 + 5 = 15 10 + 5 = 15 10 + 5 = 15 10 + 5 = 15 10 + 5 = 15 10 + 5 = 15 10 + 5 = 15 Winner 166 2 / 3 Finalists Aashish Mehrotra 8 16 18 20 8 + 4 = 12 10 + 4 = 14 8 + 4 = 12 10 + 4 = 14 10 + 5 = 15 8 + 5 = 13 10 + 4 = 14 Runner up 156 Shruti Sharma 9 18 19 18 10 + 5 = 15 10 + 4 = 14 10 + 4 = 14 10 + 4 = 14 10 + 5 = 15 10 + 5 = 15 10 + 5 = 15 Runner up 166 Harshvardhan Deo NOT IN COMPETITION Debut 8 + 4 = 12 10 + 4 = 14 10 + 4 = 14 10 + 4 = 14 Runner up 54 Naina Singh 7 15 15 18 10 + 4 = 14 7 + 3 = 10 8 + 4 = 12 10 + 4 = 14 8 + 5 = 13 10 + 5 = 15 10 + 4 = 14 Runner up 147 Semifinalists Ansh Bagri 10 19 18 17 10 + 4 = 14 7 + 4 = 11 8 + 4 = 12 8 + 4 = 12 10 + 5 = 15 10 + 5 = 15 8 + 4 = 12 ELIMINATED 155 Lekha Prajapati 7 16 16 16 8 + 4 = 12 8 + 4 = 12 8 + 4 = 12 9 + 4 = 13 10 + 4 = 14 10 + 5 = 15 10 + 4 = 14 ELIMINATED 147 5 Shariq Nanda 9 16 14 14 10 + 3 = 13 8 + 4 = 12 8 + 3 = 11 10 + 4 = 14 8 + 5 = 13 ELIMINATED 116 Elisha Mayor NOT IN COMPETITION Debut 8 + 4 = 12 8 + 5 = 13 ELIMINATED 25 6 Tapan Singh NOT IN COMPETITION Debut 6 + 3 = 9 ELIMINATED 9 Simran Choudhary NOT IN COMPETITION Debut 8 + 4 = 12 ELIMINATED 12 7 Karan Taluja 7 17 16 16 6 + 3 = 9 8 + 4 = 12 6 + 3 = 9 ELIMINATED 86 Angela Krislinzki 9 18 20 16 8 + 5 = 13 6 + 3 = 9 6 + 3 = 9 ELIMINATED 97 8 Ahmed Masih 8 14 16 18 6 + 3 = 9 6 + 4 = 10 ELIMINATED 75 Kanikka Kapur 10 19 19 19 6 + 3 = 9 6 + 3 = 9 ELIMINATED 97 9 Harshvardhan Ahlawat 7 14 12 13 6 + 3 = 9 ELIMINATED 55 Pranati Rai Prakash 7 13 17 14 6 + 3 = 9 ELIMINATED 60 10 Jash Bahl 10 14 12 ELIMINATED 36 Heli Vyas 6 12 12 ELIMINATED 30 11 Shivank Chaudhary 6 12 ELIMINATED 18 Sahiba Bhasin 7 12 ELIMINATED 19 12 Karanraj Sharma 6 ELIMINATED 6 Laveena Keswani 6 ELIMINATED 6
  Fact 0.0: The table is headed Place, Actor / Actress, Episodes, Finals, and Total score (175), with round columns labeled 3-4 through 25-26.
  Fact 0.1: Aman Gandotra's total score is 141 and he is a Winner.
  Fact 0.2: Natasha Bharadwaj's total score is 166 and she is a Winner.
  Fact 0.3: Aashish Mehrotra is a Finalist with a Runner up placement and a total score of 156.
  Fact 0.4: Shruti Sharma is a Finalist with a Runner up placement and a total score of 166.
  Fact 0.5: Harshvardhan Deo was not in the competition and made a Debut, with a Runner up placement and a total score of 54.
  Fact 0.6: Naina Singh is a Finalist with a Runner up placement and a total score of 147.
  Fact 0.7: Ansh Bagri is a Semifinalist who was ELIMINATED with a total score of 155.
  Fact 0.8: Lekha Prajapati is a Semifinalist who was ELIMINATED with a total score of 147.
  Fact 0.9: Shariq Nanda was ELIMINATED with a total score of 116.
  Fact 0.10: Elisha Mayor was not in the competition, made a Debut, was ELIMINATED, and has a total score of 25.
  Fact 0.11: Tapan Singh was not in the competition, made a Debut, was ELIMINATED, and has a total score of 9.
  Fact 0.12: Simran Choudhary was not in the competition, made a Debut, was ELIMINATED, and has a total score of 12.
  Fact 0.13: Karan Taluja was ELIMINATED with a total score of 86.
  Fact 0.14: Angela Krislinzki was ELIMINATED with a total score of 97.
  Fact 0.15: Ahmed Masih was ELIMINATED with a total score of 75.
  Fact 0.16: Kanikka Kapur was ELIMINATED with a total score of 97.
  Fact 0.17: Harshvardhan Ahlawat was ELIMINATED with a total score of 55.
  Fact 0.18: Pranati Rai Prakash was ELIMINATED with a total score of 60.
  Fact 0.19: Jash Bahl was ELIMINATED with a total score of 36.
  Fact 0.20: Heli Vyas was ELIMINATED with a total score of 30.
  Fact 0.21: Shivank Chaudhary was ELIMINATED with a total score of 18.
  Fact 0.22: Sahiba Bhasin was ELIMINATED with a total score of 19.
  Fact 0.23: Karanraj Sharma was ELIMINATED with a total score of 6.
  Fact 0.24: Laveena Keswani was ELIMINATED with a total score of 6.
Sentence 1: India 's Next Superstars is a talent - searcher Indian reality show , which premiered on Star Plus and is streamed on Hotstar . Karan Johar and Rohit Shetty are the judges for the show . Aman Gandotra and Natasha Bharadwaj are declared winners of 2018 season , alongside Shruti Sharma who won a ' Special Mention ' award . Runners up in the male category were Aashish Mehrotra and Harshvardhan Deo and in the female category were Naina Singh and Shruti Sharma .
  Fact 1.0: India's Next Superstars is an Indian talent-searcher reality show.
  Fact 1.1: India's Next Superstars premiered on Star Plus.
  Fact 1.2: India's Next Superstars is streamed on Hotstar.
  Fact 1.3: Karan Johar and Rohit Shetty are the judges for India's Next Superstars.
  Fact 1.4: Aman Gandotra and Natasha Bharadwaj are declared winners of the 2018 season.
  Fact 1.5: Shruti Sharma is declared a winner of the 2018 season alongside Aman Gandotra and Natasha Bharadwaj.
  Fact 1.6: Shruti Sharma won a 'Special Mention' award.
  Fact 1.7: Aashish Mehrotra and Harshvardhan Deo were the runners up in the male category.
  Fact 1.8: Naina Singh and Shruti Sharma were the runners up in the female category.

Retrieved items:
<item 1>
*India’s Next Superstars* was a Star Plus acting talent reality show, streamed on Hotstar from January 13 to April 8, 2018. Produced by Dharma Productions and
Endemol Shine India, it followed 20 contestants—and later wildcards—training and competing through acting, dance and themed challenges. Hosts Karan Wahi and
Rithvik Dhanjani presented performances judged by Karan Johar and Rohit Shetty; Mukesh Chhabra and
</item 1>

<item 2>
*India’s Next Superstars* was a 2018 Indian Hindi-language acting talent reality show on Star Plus. Hosted by Rithvik Dhanjani and Karan Wahi, it featured
judges Rohit Shetty and Karan Johar, with Mukesh Chhabra and Mahesh Bhatt judging its “INS Pathshala” segment. Aman Gandotra, Natasha Bharadwaj and Shruti
Sharma were declared winners.
</item 2>

<item 3>
India's Next Superstars
Contents ( hide ) 1 Concept 2 Contestants 3 Guest appearance 4 Scoring chart 5 Synopsis of Episodes 6 Lights Camera Superstar cocktails 7 Superstars Ki
Paathshaala 8 References 9 External links Concept ( edit ) The new talent hunt show of Karan Johar and Rohit Shetty is concerned , ten males and ten females
will be living under one roof , and every day they will be assigned for tasks and will be judged on the basis of their performances . Additionally , two more
males and females , will join the competition as Wild - Cards . The two winners will be rewarded with a three - film deal with Dharma Productions house .
Contestants ( edit ) BOYS Names Notes Aashish Mehrotra RUNNER UP Ahmed Masih ELIMINATED ON 25 FEBRUARY 2018 Aman Gandotra WINNER Ansh Bagri SEMIFINALIST ;
</item 3>

<item 4>
India’s Next Superstars, hosted by Karan Wahi and Rithvik Dhanjani, pairs aspiring actors for performances judged by filmmakers Karan Johar and Rohit Shetty,
with celebrity guest Priyanka Chopra at the launch. Contestants reimagined what happened after famous films, including Badrinath Ki Dulhania, Rab Ne Bana Di
Jodi, Border and DDLJ; later themes included jealousy, romance and sacrifice. Judges praised standout acting, dancing
</item 4>

<item 5>
*India’s Next Superstars* was a Star Plus talent-search reality show, also streamed on Hotstar, that ran from 13 January to 8 April 2018. Dharma Productions and
Endemol Shine India produced it. Judges Karan Johar and Rohit Shetty assessed contestants’ daily tasks. Twenty contestants—ten men and ten women—lived together,
with additional wild cards joining; the two winners earned a three-film deal with Dharma Productions. Aman Gand
</item 5>

<item 6>
The excerpts follow *India’s Next Superstars*, where contestants performed Bollywood-inspired comedy, romance, dance and emotional dramas, earning scores,
spotlight honors and occasional eliminations. Judges Karan Johar, Rohit Shetty and others praised standout portrayals—including Chaplin, Sita, Rani Lakshmibai
and Sania Mirza—while offering acting feedback. Acts addressed disability, gender discrimination, women’s empowerment, cancer and family hardship; contestants
also shared personal struggles. The competition shifted to
</item 6>

<item 7>
*India’s Next Superstars* crowned Natasha Bharadwaj winner and named Naina Singh runner-up; Lekha Prajapati was a semifinalist. The contestant lists include
boys and girls eliminated from January through April 2018, with several wildcard entrants on 3 March. Harshvardhan Deo entered as a wildcard and finished second
runner-up; Tapan Singh and Simran Choudhary were also wildcard entrants. Shruti Sharma is inconsistently described
</item 7>

<item 8>
India's Next Superstars
India 's next Superstars - Wikipedia India 's next Superstars Jump to : navigation , search This article may require copy editing for MOS issues (
Capitalization , bold , etc . ) . You can assist by editing it . ( April 2018 ) ( Learn how and when to remove this template message ) India 's Next Superstars
Genre Reality Based on Acting Presented by Rithvik Dhanjani Karan Wahi Judges INS Judges : Rohit Shetty Karan Johar INS Pathshala Judges : Mukesh Chhabra Mahesh
Bhatt Country of origin India No. of seasons No.
</item 8>

<item 9>
India’s Next Superstars premiered on 13 January 2018, training 20 contestants in acting, dance and other film skills under judges Karan Johar and Rohit Shetty,
hosted by Rithvik Dhanjani and Karan Wahi. Celebrity guests included Priyanka Chopra, Sidharth Malhotra, Kangana Ranaut, Tiger Shroff and Badshah. Contestants
faced themed performances, scores and eliminations; four wildcard entrants joined in
</item 9>

<item 10>
India's Next Superstars
of seasons No. of episodes INS : 25 ( as of 7 April 2018 ) INS Pathshala : 60 ( as of 6 April 2018 ) Production Running time 1 hr 30 mins Production company ( s
) Dharma Productions Endemol Shine India Distributor Star India Release Original network STAR Plus Picture format 720p SDTV 1080p HDTV Original release 13
January ( 2018 - 01 - 13 ) -- 8 April 2018 External links Hotstar India 's Next Superstars is a talent - searcher Indian reality show , which premiered on Star
Plus and is streamed on Hotstar . Karan Johar and Rohit Shetty are the judges for the show . Aman Gandotra and Natasha Bharadwaj are declared winners of 2018
season , alongside Shruti Sharma who won a ' Special Mention ' award . Runners up in the male category were Aashish Mehrotra and Harshvardhan Deo and in the
female category were Naina Singh and Shruti Sharma .
</item 10>
=== END ITEM id=81abcd530d79365a ===

=== ITEM id=963aed7dc37dac68 ===
Gold sentences and facts:
Sentence 0: The Eagles and the Patriots met again in Super Bowl LII , following the 2017 season , with the Eagles taking their revenge 41 -- 33 .
  Fact 0.0: The Eagles and the Patriots met again in Super Bowl LII.
  Fact 0.1: The Eagles and the Patriots met in Super Bowl LII following the 2017 season.
  Fact 0.2: The Eagles beat the Patriots 41 to 33 in Super Bowl LII.
  Fact 0.3: The Eagles took revenge on the Patriots in Super Bowl LII.

Retrieved items:
<item 1>
Philadelphia Eagles
As of the conclusion of the 2016 season , during the Lurie era , the Eagles are 1 -- 4 in conference championship games and 0 -- 1 in Super Bowls . On August 13
, 2009 , the Eagles signed quarterback Michael Vick . On December 6 , 2009 , Andy Reid became only the 5th coach in NFL history to win 100 or more games with a
single team in a single decade . The other four are Tom Landry , Don Shula , Tony Dungy , and Bill Belichick , all Super Bowl winners . McNabb finally had a
complete receiving corps , between first round draft pick Jeremy Maclin , DeSean Jackson 's 1,000 yard season , and Brent Celek ranking among the top 5 tight
ends in the league . Without Brian Dawkins , defensive end Trent Cole stepped up and became the dominant force on defense with 12 sacks , earning him his second
trip to the Pro Bowl and All - Pro honors .
</item 1>

<item 2>
Super Bowl XXXIX
For the Eagles , with the loss , the city of Philadelphia would not see its first championship since the 76ers swept the 1983 NBA Finals until the next time
their city 's teams played for a championship , when the Phillies won the 2008 World Series . The Eagles did not return to the Super Bowl until 2017 , also with
a 13 - 3 record , and defeating the Falcons and Vikings in the playoffs . The Eagles and the Patriots met again in Super Bowl LII , following the 2017 season ,
with the Eagles taking their revenge 41 -- 33 . This is the last time that a team has won back - to - back Super Bowls , with the Patriots losing their second
playoff game a year later and then , for the next eight years , every team either losing their first playoff game or missing them altogether . The streak was
broken in 2014 when the Seahawks defeated the Panthers and ultimately advanced to the Super Bowl , only to lose to a Patriots goal line stand .
</item 2>

<item 3>
Super Bowl XXXIX
New England also still had their trio of impact veteran linebackers : Pro Bowler Tedy Bruschi ( 122 tackles , 3.5 sacks , 3 interceptions , and 70 return yards
) , Willie McGinest ( 9.5 sacks , 1 fumble recovery , and 1 interception ) , and Mike Vrabel ( 71 tackles and 5.5 sacks ) , along with Ted Johnson . Vrabel also
frequently played at the tight end position during offensive plays near the opponent 's goal line , recording 2 touchdown receptions . Eagles return to the
Super Bowl ( edit ) Main article : 2004 Philadelphia Eagles season The Eagles gained the 2004 NFC Super Bowl berth after 3 consecutive defeats in the NFC
Championship Game to the St. Louis Rams , Tampa Bay Buccaneers ( who won Super Bowl XXXVII that year ) , and Carolina Panthers , respectively . The Eagles hired
Andy Reid as their head coach in 1999 following two straight losing seasons . That same year , they used their first - round pick in the NFL draft ( the second
overall ) to select quarterback Donovan McNabb .
</item 3>

<item 4>
Philadelphia Eagles
In 2003 , they won the NFC first seed , but Westbrook went down in Week 17 , culminating in a loss to the Carolina Panthers in their 3rd straight NFC
Championship . 2004 season and Super Bowl XXXIX In 2004 , the Philadelphia Eagles had their best season since 1960 , going 13 -- 1 before resting their starters
and losing their next 2 , clinching the 1st seed for the second year in a row . McNabb set career highs , completing 64 % of his passes for 3,875 yards , though
he did n't play all 16 games . McNabb became the first quarterback ever to throw more than 30 touchdowns and fewer than 10 interceptions in a season . His
success could be attributed to the fact that he had a reliable receiver , Terrell Owens , who got 1,200 yards and 14 touchdowns in 14 games . After beating the
Vikings and Falcons the Eagles advanced to Super Bowl XXXIX , where they dueled the New England Patriots .
</item 4>

<item 5>
Super Bowl XXXIX
The Eagles were trying to win their first NFL title since 1960 and the first championship for the city of Philadelphia since Moses Malone 's `` fo ' , fi ' , fo
' '' during the 76ers run to the 1983 NBA Championship . With President George W. Bush being inaugurated for a second time in January , the Eagles were also
trying to end a losing streak -- teams in the city of Philadelphia had lost six straight championships during presidential inauguration years , beginning with
the 76ers loss in 1977 . The streak included the Eagles in Super Bowl XV in 1981 . 1989 was not included in that streak , but 1977 , 1981 , 1985 ( Flyers ) ,
1993 ( Phillies ) , 1997 ( Flyers ) , and 2001 ( 76ers ) were . Television and entertainment ( edit ) The game was televised in the United States by Fox , with
play - by - play announcer Joe Buck and color commentators Cris Collinsworth and Troy Aikman . Pam Oliver ( Patriots sideline ) and Chris Myers ( Eagles
sideline ) served as sideline reporters .
</item 5>

<item 6>
Philadelphia Eagles
Foles had his best game since week 15 and threw for 352 passing yards and 3 touchdowns including an impressive flea flicker touchdown pass . This win marked the
franchise 's third Super Bowl appearance and a berth in Super Bowl LII for a 2005 rematch against Tom Brady and the New England Patriots . With Foles at the
helm , the Eagles started off the game strong , leading the Patriots 22 -- 12 at halftime . In the second half the Patriots gained momentum and for the first
time took the lead in the 4th quarter 33 -- 32 . The Eagles rallied back and scored an 11 - yard Touchdown to tight end Zach Ertz . Later the Eagles converted
on a 46 - yard field goal by Jake Elliott to lead 41 -- 33 . On the final play of the game with 9 seconds left , Brady threw a hail - mary pass directed toward
tight end Rob Gronkowski which was bobbled by Patrick Robinson before being dropped in the end - zone as time expired , securing the win for the Eagles 41 -- 33
, and ending the 58 - year long championship drought .
</item 6>

<item 7>
Graph query row:
team: Philadelphia Eagles
super_bowl: Super Bowl LII
date_played: 2018
result: None
outcome: None
</item 7>

<item 8>
Philadelphia Eagles
As a result , the Eagles and Steelers no longer played each other every year . Currently they are scheduled to meet once every four years in the regular season
, the most recent meeting being in 2016 at Philadelphia , with the Eagles winning 34 - 3 . The Steelers have lost nine straight games on the road against the
Eagles dating back to 1966 , which was also the start of the Super Bowl era . The Eagles lead the all - time series 47 -- 28 -- 3 . Logo and uniforms See also :
Uniform ( American football ) and footnote about Eagles ' uniform numbers . For several decades , the Eagles ' colors were kelly green , silver , and white . In
1954 , the Eagles , along with the Baltimore Colts , became the second team ever in the NFL to put a logo on their helmets , with silver wings on a kelly green
helmet . In 1969 , the team wore two helmet versions : Kelly green with white wings in road games , and white with kelly green wings at home . From 1970 to ' 73
, they wore the white helmets with Kelly green wings exclusively before switching back to Kelly green helmets with silver wings .
</item 8>

<item 9>
Philadelphia Eagles
Season records Main article : List of Philadelphia Eagles seasons Regular season record ( all - time ) : 568 -- 594 -- 26 Playoff record ( all - time ) : 22 --
21 ( last appearance after beating New England Patriots in Super Bowl LII ) Most points in a season : 474 points ( 2014 ) NFL championships won : 4 ( 3 before
the 1967 NFL - AFL merger created the Super Bowl ) Super Bowls won : 1 out of 3 appearances Passing leader ( all - time ) : Donovan McNabb -- 32,873 yards
Rushing leader ( all - time ) : LeSean McCoy -- 6,792 yards Receiving leader ( all - time ) : Harold Carmichael -- 8,978 yards Winningest coach ( all - time ) :
Andy Reid -- 130 wins Top player by approximate value ( all - time ) : Donovan McNabb -- 126 AV Rivalries New York Giants Main article : Eagles -- Giants
rivalry One of the NFL 's oldest , this rivalry began on October 15 , 1933 when the Giants defeated the newly founded Eagles 56 -- 0 .
</item 9>

<item 10>
Philadelphia Eagles
2002 , 2003 , 2004 , 2006 , 2008 , 2009 , 2010 , 2013 , 2017 Home fields Baker Bowl ( 1933 -- 1935 ) Philadelphia Municipal Stadium ( 1936 -- 1939 , 1941 )
Connie Mack Stadium ( 1940 , 1942 -- 1957 ) Franklin Field ( 1958 -- 1970 ) Veterans Stadium ( 1971 -- 2002 ) Lincoln Financial Field ( 2003 -- present ) The
Philadelphia Eagles are a professional American football franchise based in Philadelphia , Pennsylvania . The Eagles compete in the National Football League (
NFL ) as a member club of the league 's National Football Conference ( NFC ) East division . They are Super Bowl champions , having won Super Bowl LII , their
fourth NFL title , after winning in 1948 , 1949 , and 1960 . The franchise was established in 1933 as a replacement for the bankrupt Frankford Yellow Jackets ,
when a group led by Bert Bell secured the rights to an NFL franchise in Philadelphia .
</item 10>
=== END ITEM id=963aed7dc37dac68 ===

=== ITEM id=7266e0ceb4c17edf ===
Gold sentences and facts:
Sentence 0: Economy of India Mumbai , Maharashtra the financial center of India Currency Indian rupee ( INR ) ₹ 1 = 100 Paise Fiscal year 1 April -- 31 March Trade organisations WTO , WCO , WFTU , G - 20 and others Statistics GDP $2.848 trillion ( nominal ; 2018 est ) $10.385 trillion ( PPP ; 2018 est ) GDP rank 6th ( nominal ) 3rd ( PPP ) GDP growth 7.2 % ( Q3 , 2017 - 18 ) ( MOSPI ) GDP per capita $2,134 ( nominal ; 2018 est ) $7,783 ( PPP ; 2018 est ) GDP per capita rank 141st ( nominal ) 123rd ( PPP ) GDP by sector Agriculture : 17.32 % Industry : 29.02 % Services : 53.66 % ( 2016 est . ) Inflation ( CPI ) 4.28 % ( March 2018 ) Base borrowing rate 6.25 % ( as on 16 Mar. 2018 ) Population below poverty line 21.2 % 58 % live less than $3.10 / day ( 2011 ; World bank estimate ) Gini coefficient 33.9 ( 2013 ) Human Development Index 0.624 ( 2015 ) medium ( 131st ) Labour force 520.2 million ( 2017 ) Labour force by occupation Agriculture : 47 % Industry : 22 % Services : 31 % ( FY 2014 est . ) Unemployment 8.8 % ( 2017 ) Average net salary ₹ 82,269 / $1,284 annually ( 2016 - 17 ) Main industries Textiles chemicals food processing steel cement mining petroleum machinery software pharmaceuticals transportation equipment Ease - of - doing - business rank 100 ( 2017 ) External Exports $275.8 billion ( 2017 ) Export goods Agricultural products 13.2 % Fuels and mining products 15.7 % Manufacturers 68.4 % Others 2.7 % Main export partners European Union 17.6 % United States 16.1 % United Arab Emirates 11.5 % Hong Kong 5.1 % China 3.4 % Other 46.3 % Imports $384.3 billion ( 2017 ) Import goods Agricultural products 7.1 % Fuels and mining products 33.1 % Manufacturers 47.8 % Other 12 % Main import partners China 17 % European Union 11.3 % United States 5.7 % United Arab Emirates 5.4 % Saudi Arabia 5.2 % Other 55.5 % FDI stock Inward : $318.50 billion Outward : $144.13 billion ( 2016 ) Current account 0.7 % of GDP ( 2016 -- 17 ) Gross external debt $471.9 billion ( 31 March 2017 ) Net international investment position - $404.8 billion ( Sept. 2017 ) Public finances Public debt 67.7 % of GDP ( 2017 ) Revenues ₹ 35.41 trillion ( US $540 billion ) ( 2017 ) Expenses ₹ 46.25 trillion ( US $710 billion ) ( 2017 ) Economic aid $2.68 billion ( 2016 ) Credit rating Standard & Poor 's : BBB − ( Domestic ) BBB − ( Foreign ) BBB+ ( T&C Assessment ) Outlook : Stable Moody 's : Baa2 Outlook : Stable Fitch : BBB − Outlook : Stable Foreign reserves $426.082 billion ( 13 April 2018 ) ( 8th ) Main data source : CIA World Fact Book All values , unless otherwise stated , are in US dollars .
  Fact 0.0: Mumbai, Maharashtra is the financial center of India.
  Fact 0.1: The currency of India is the Indian rupee (INR).
  Fact 0.2: 1 rupee equals 100 paise.
  Fact 0.3: India's fiscal year runs from 1 April to 31 March.
  Fact 0.4: India's trade organisations include WTO, WCO, WFTU, G-20 and others.
  Fact 0.5: India's nominal GDP is $2.848 trillion (2018 estimate).
  Fact 0.6: India's PPP GDP is $10.385 trillion (2018 estimate).
  Fact 0.7: India's nominal GDP rank is 6th.
  Fact 0.8: India's PPP GDP rank is 3rd.
  Fact 0.9: India's GDP growth is 7.2% (Q3, 2017-18), per MOSPI.
  Fact 0.10: India's nominal GDP per capita is $2,134 (2018 estimate).
  Fact 0.11: India's PPP GDP per capita is $7,783 (2018 estimate).
  Fact 0.12: India's nominal GDP per capita rank is 141st.
  Fact 0.13: India's PPP GDP per capita rank is 123rd.
  Fact 0.14: Agriculture makes up 17.32% of India's GDP by sector (2016 estimate).
  Fact 0.15: Industry makes up 29.02% of India's GDP by sector (2016 estimate).
  Fact 0.16: Services make up 53.66% of India's GDP by sector (2016 estimate).
  Fact 0.17: India's CPI inflation is 4.28% (March 2018).
  Fact 0.18: India's base borrowing rate is 6.25% (as on 16 March 2018).
  Fact 0.19: 21.2% of India's population is below the poverty line.
  Fact 0.20: 58% of India's population lives on less than $3.10 per day (2011, World Bank estimate).
  Fact 0.21: India's Gini coefficient is 33.9 (2013).
  Fact 0.22: India's Human Development Index is 0.624 (2015), rated medium and ranked 131st.
  Fact 0.23: India's labour force is 520.2 million (2017).
  Fact 0.24: Agriculture accounts for 47% of India's labour force by occupation (FY 2014 estimate).
  Fact 0.25: Industry accounts for 22% of India's labour force by occupation (FY 2014 estimate).
  Fact 0.26: Services account for 31% of India's labour force by occupation (FY 2014 estimate).
  Fact 0.27: India's unemployment rate is 8.8% (2017).
  Fact 0.28: India's average net salary is ₹82,269 or $1,284 annually (2016-17).
  Fact 0.29: Main industries in India include textiles, chemicals, food processing, steel, cement, mining, petroleum, machinery, software, pharmaceuticals and transportation equipment.
  Fact 0.30: India's ease-of-doing-business rank is 100 (2017).
  Fact 0.31: India's exports are $275.8 billion (2017).
  Fact 0.32: Agricultural products make up 13.2% of India's export goods.
  Fact 0.33: Fuels and mining products make up 15.7% of India's export goods.
  Fact 0.34: Manufacturers make up 68.4% of India's export goods.
  Fact 0.35: Other export goods make up 2.7% of India's exports.
  Fact 0.36: The European Union is 17.6% of India's main export partners.
  Fact 0.37: The United States is 16.1% of India's main export partners.
  Fact 0.38: The United Arab Emirates is 11.5% of India's main export partners.
  Fact 0.39: Hong Kong is 5.1% of India's main export partners.
  Fact 0.40: China is 3.4% of India's main export partners.
  Fact 0.41: Other export partners make up 46.3% of India's exports.
  Fact 0.42: India's imports are $384.3 billion (2017).
  Fact 0.43: Agricultural products make up 7.1% of India's import goods.
  Fact 0.44: Fuels and mining products make up 33.1% of India's import goods.
  Fact 0.45: Manufacturers make up 47.8% of India's import goods.
  Fact 0.46: Other import goods make up 12% of India's imports.
  Fact 0.47: China is 17% of India's main import partners.
  Fact 0.48: The European Union is 11.3% of India's main import partners.
  Fact 0.49: The United States is 5.7% of India's main import partners.
  Fact 0.50: The United Arab Emirates is 5.4% of India's main import partners.
  Fact 0.51: Saudi Arabia is 5.2% of India's main import partners.
  Fact 0.52: Other import partners make up 55.5% of India's imports.
  Fact 0.53: India's inward FDI stock is $318.50 billion (2016).
  Fact 0.54: India's outward FDI stock is $144.13 billion (2016).
  Fact 0.55: India's current account is 0.7% of GDP (2016-17).
  Fact 0.56: India's gross external debt is $471.9 billion (31 March 2017).
  Fact 0.57: India's net international investment position is -$404.8 billion (Sept. 2017).
  Fact 0.58: India's public debt is 67.7% of GDP (2017).
  Fact 0.59: India's public revenues are ₹35.41 trillion, or US$540 billion (2017).
  Fact 0.60: India's public expenses are ₹46.25 trillion, or US$710 billion (2017).
  Fact 0.61: India's economic aid is $2.68 billion (2016).
  Fact 0.62: Standard & Poor's gives India a BBB- domestic credit rating.
  Fact 0.63: Standard & Poor's gives India a BBB- foreign credit rating.
  Fact 0.64: Standard & Poor's gives India a BBB+ T&C assessment.
  Fact 0.65: Standard & Poor's outlook for India is stable.
  Fact 0.66: Moody's gives India a Baa2 rating with a stable outlook.
  Fact 0.67: Fitch gives India a BBB- rating with a stable outlook.
  Fact 0.68: India's foreign reserves are $426.082 billion (13 April 2018), ranked 8th.
  Fact 0.69: The main data source for these figures is the CIA World Fact Book.
  Fact 0.70: All values, unless otherwise stated, are in US dollars.
Sentence 1: India has one of the fastest growing service sectors in the world with an annual growth rate above 9 % since 2001 , which contributed to 57 % of GDP in 2012 -- 13 . India has become a major exporter of IT services , Business Process Outsourcing ( BPO ) services , and software services with $154 billion revenue in FY 2017 . This is the fastest - growing part of the economy . The IT industry continues to be the largest private - sector employer in India . India is the third - largest start - up hub in the world with over 3,100 technology start - ups in 2014 -- 15 The agricultural sector is the largest employer in India 's economy but contributes to a declining share of its GDP ( 17 % in 2013 -- 14 ) . India ranks second worldwide in farm output . The industry ( manufacturing ) sector has held a steady share of its economic contribution ( 26 % of GDP in 2013 -- 14 ) . The Indian automobile industry is one of the largest in the world with an annual production of 21.48 million vehicles ( mostly two and three - wheelers ) in 2013 -- 14 . India had $600 billion worth of retail market in 2015 and one of world 's fastest growing e-commerce markets .
  Fact 1.0: India has one of the fastest growing service sectors in the world.
  Fact 1.1: India's service sector has grown at an annual rate above 9% since 2001.
  Fact 1.2: India's service sector contributed 57% of GDP in 2012-13.
  Fact 1.3: India has become a major exporter of IT services, Business Process Outsourcing (BPO) services and software services.
  Fact 1.4: India's IT, BPO and software services exports had $154 billion in revenue in FY 2017.
  Fact 1.5: The IT and services export part of the economy is the fastest-growing part of the economy.
  Fact 1.6: The IT industry is the largest private-sector employer in India.
  Fact 1.7: India is the third-largest start-up hub in the world.
  Fact 1.8: India had over 3,100 technology start-ups in 2014-15.
  Fact 1.9: The agricultural sector is the largest employer in India's economy.
  Fact 1.10: Agriculture's share of India's GDP is declining, at 17% in 2013-14.
  Fact 1.11: India ranks second worldwide in farm output.
  Fact 1.12: The industry (manufacturing) sector has held a steady share of India's economic contribution, at 26% of GDP in 2013-14.
  Fact 1.13: The Indian automobile industry is one of the largest in the world.
  Fact 1.14: The Indian automobile industry produced 21.48 million vehicles in 2013-14, mostly two- and three-wheelers.
  Fact 1.15: India had a $600 billion retail market in 2015.
  Fact 1.16: India has one of the world's fastest growing e-commerce markets.
Sentence 2: India ranks second worldwide in farm output . Agriculture and allied sectors like forestry , logging and fishing accounted for 17 % of the GDP and employed 49 % of its total workforce in 2014 . Agriculture accounted for 23 % of GDP , and employed 59 % of the country 's total workforce in 2016 . As the Indian economy has diversified and grown , agriculture 's contribution to GDP has steadily declined from 1951 to 2011 , yet it is still the country 's largest employment source and a significant piece of its overall socio - economic development . Crop - yield - per - unit - area of all crops has grown since 1950 , due to the special emphasis placed on agriculture in the five - year plans and steady improvements in irrigation , technology , application of modern agricultural practices and provision of agricultural credit and subsidies since the Green Revolution in India . However , international comparisons reveal the average yield in India is generally 30 % to 50 % of the highest average yield in the world . The states of Uttar Pradesh , Punjab , Haryana , Madhya Pradesh , Andhra Pradesh , Telangana , Bihar , West Bengal , Gujarat and Maharashtra are key contributors to Indian agriculture .
  Fact 2.0: India ranks second worldwide in farm output.
  Fact 2.1: Agriculture and allied sectors, such as forestry, logging and fishing, accounted for 17% of India's GDP in 2014.
  Fact 2.2: Agriculture and allied sectors employed 49% of India's total workforce in 2014.
  Fact 2.3: Agriculture accounted for 23% of India's GDP in 2016.
  Fact 2.4: Agriculture employed 59% of India's total workforce in 2016.
  Fact 2.5: Agriculture's contribution to India's GDP has steadily declined from 1951 to 2011.
  Fact 2.6: Agriculture is still the country's largest employment source.
  Fact 2.7: Agriculture is a significant piece of India's overall socio-economic development.
  Fact 2.8: Crop yield per unit area of all crops in India has grown since 1950.
  Fact 2.9: Crop yield growth in India is due to the special emphasis on agriculture in the five-year plans.
  Fact 2.10: Crop yield growth in India is due to steady improvements in irrigation and technology.
  Fact 2.11: Crop yield growth in India is due to application of modern agricultural practices.
  Fact 2.12: Crop yield growth in India is due to provision of agricultural credit and subsidies since the Green Revolution.
  Fact 2.13: International comparisons reveal that the average yield in India is generally 30% to 50% of the highest average yield in the world.
  Fact 2.14: Uttar Pradesh, Punjab, Haryana, Madhya Pradesh, Andhra Pradesh, Telangana, Bihar, West Bengal, Gujarat and Maharashtra are key contributors to Indian agriculture.

Retrieved items:
<item 1>
Economy of India
Agriculture and allied sectors like forestry , logging and fishing accounted for 17 % of the GDP and employed 49 % of its total workforce in 2014 . Agriculture
accounted for 23 % of GDP , and employed 59 % of the country 's total workforce in 2016 . As the Indian economy has diversified and grown , agriculture 's
contribution to GDP has steadily declined from 1951 to 2011 , yet it is still the country 's largest employment source and a significant piece of its overall
socio - economic development . Crop - yield - per - unit - area of all crops has grown since 1950 , due to the special emphasis placed on agriculture in the
five - year plans and steady improvements in irrigation , technology , application of modern agricultural practices and provision of agricultural credit and
subsidies since the Green Revolution in India . However , international comparisons reveal the average yield in India is generally 30 % to 50 % of the highest
average yield in the world . The states of Uttar Pradesh , Punjab , Haryana , Madhya Pradesh , Andhra Pradesh , Telangana , Bihar , West Bengal , Gujarat and
Maharashtra are key contributors to Indian agriculture .
</item 1>

<item 2>
Agriculture in India
Bihar Peasant Life . Bengal Secretariat Press , Calcutta . References ( edit ) Jump up ^ https://books.google.com/books?id=pU6dCgAAQBAJ&pg=PA96 Jump up ^
Agriculture 's share in GDP declines to 13.7 % in 2012 - 13 Jump up ^ `` CIA Factbook : India '' . CIA Factbook . Central Intelligence Agency . Archived from
the original on 11 June 2008 . Retrieved 2008 - 06 - 10 . Jump up ^ Staff , India Brand Equity Foundation Agriculture and Food in India Accessed 7 May 2013 ^
Jump up to : India 's Agricultural Exports Climb to Record High , United States Department of Agriculture ( 2014 ) Jump up ^
http://www.ibef.org/exports/agriculture-and-food-industry-india.aspx Jump up ^ http://www.agriexam.com/?m=1 ^ Jump up to : `` FAOSTAT , 2010 data '' .
Faostat.fao.org . Retrieved 2011 - 09 - 17 .
</item 2>

<item 3>
Agriculture in India
Contents ( hide ) 1 Overview 2 History 2.1 Agriculture and colonialism 2.2 Indian agriculture after independence 3 Irrigation 4 Output 4.1 Major crops and
yields 4.2 Horticulture 4.3 Organic Agriculture 5 Problems 5.1 Infrastructure 5.2 Productivity 5.3 Farmer suicides 5.4 Diversion of agricultural land for non
agricultural purpose 6 Initiatives 7 Maps 8 See also 9 Bibliography 10 References 11 Further reading 12 External links Overview ( edit ) Rice Paddy Fields in
Tamil Nadu As per the 2010 FAO world agriculture statistics India is the world 's largest producer of many fresh fruits and vegetables , milk , major spices ,
select fibrous crops such as jute , staples such as millets and castor oil seed . India is the second largest producer of wheat and rice , the world 's major
food staples . India is the world 's second or third largest producer of several dry fruits , agriculture - based textile raw materials , roots and tuber crops
, pulses , farmed fish , eggs , coconut , sugarcane and numerous vegetables .
</item 3>

<item 4>
Economy of India
Employment See also : Labour in India , Indian labour law , Child labour in India , and Unemployment in India Agricultural and allied sectors accounted for
about 52.1 % of the total workforce in 2009 -- 10 . While agriculture employment has fallen over time in percentage of labour employed , services which includes
construction and infrastructure have seen a steady growth accounting for 20.3 % of employment in 2012 -- 13 . Of the total workforce , 7 % is in the organised
sector , two - thirds of which are in the government - controlled public sector . About 51.2 % of the workforce in India is self - employed . According to a
2005 -- 06 survey , there is a gender gap in employment and salaries . In rural areas , both men and women are primarily self - employed , mostly in agriculture
. In urban areas , salaried work was the largest source of employment for both men and women in 2006 . Unemployment in India is characterised by chronic (
disguised ) unemployment .
</item 4>

<item 5>
Agriculture in India
The economic contribution of agriculture to India 's GDP is steadily declining with the country 's broad - based economic growth . Still , agriculture is
demographically the broadest economic sector and plays a significant role in the overall socio - economic fabric of India . India exported $38 billion worth of
agricultural products in 2013 , making it the seventh largest agricultural exporter worldwide and the sixth largest net exporter . Most of its agriculture
exports serve developing and least developed nations . Indian agricultural / horticultural and processed foods are exported to more than 120 countries ,
primarily in the Middle East , Southeast Asia , SAARC countries , the EU and the United States .
</item 5>

<item 6>
Economy of India
India exports several agriculture products , such as Basmati rice , wheat , cereals , spices , fresh fruits , dry fruits , buffalo beef meat , cotton , tea ,
coffee and other cash crops particularly to the Middle East , Southeast and East Asian countries . About 10 percent of its export earnings come from this trade
. Industry Industry accounts for 26 % of GDP and employs 22 % of the total workforce . According to the World Bank , India 's industrial manufacturing GDP
output in 2015 was 6th largest in the world on current US dollar basis ( $559 billion ) , and 9th largest on inflation - adjusted constant 2005 US dollar basis
( $197.1 billion ) . The industrial sector underwent significant changes due to the 1991 economic reforms , which removed import restrictions , brought in
foreign competition , led to the privatisation of certain government - owned public - sector industries , liberalised the foreign direct investment ( FDI )
regime , improved infrastructure and led to an expansion in the production of fast - moving consumer goods . Post-liberalisation , the Indian private sector was
faced with increasing domestic and foreign competition , including the threat of cheaper Chinese imports .
</item 6>

<item 7>
Agriculture in India
India 's agricultural economy is undergoing structural changes . Between 1970 and 2011 , the GDP share of agriculture has fallen from 43 % to 16 % . This is n't
because of reduced importance of agriculture or a consequence of agricultural policy . This is largely because of the rapid economic growth in services ,
industrial output , and non-agricultural sectors in India between 2000 and 2010 . Agricultural scientist MS Swaminathan has played a vital role in the green
revolution . In 2013 NDTV awarded him as 25 living legend of India for outstanding contribution to agriculture and making India a food sovereign country . An
irrigation canal in Gujarat . Irrigation contributes significantly to agriculture in India . Two states , Sikkim and Kerala have planned to shift fully to
organic farming by 2015 and 2016 respectively . Irrigation ( edit ) Main article : Irrigation in India Indian irrigation infrastructure includes a network of
major and minor canals from rivers , groundwater well - based systems , tanks , and other rainwater harvesting projects for agricultural activities . Of these ,
the groundwater system is the largest .
</item 7>

<item 8>
Economy of India
The strong efforts taken by the Department of Industrial Policy and Promotion ( DIPP ) to boost ease of doing business rankings at the state level is said to
impact the overall rankings of India . Sectors Percent labour employment in India by economic sectors ( 2010 ) . Contribution to GDP of India by economic
sectors of Indian economy have evolved between 1951 and 2013 , as its economy has diversified and developed . Historically , India has classified and tracked
its economy and GDP in three sectors : agriculture , industry and services . Agriculture includes crops , horticulture , milk and animal husbandry , aquaculture
, fishing , sericulture , aviculture , forestry and related activities . Industry includes various manufacturing sub-sectors . India 's definition of services
sector includes its construction , retail , software , IT , communications , hospitality , infrastructure operations , education , health care , banking and
insurance , and many other economic activities . Agriculture Rice fields near Puri , Odisha on India 's east coast Main articles : Agriculture in India ,
Forestry in India , Animal husbandry in India , Fishing in India , and Natural resources in India India ranks second worldwide in farm output .
</item 8>

<item 9>
Economy of India
India will soon have the largest and youngest workforce the world has ever seen . At the same time , the country is in the midst of a massive wave of
urbanization as some 10 million people move to towns and cities each year in search of jobs and opportunity . It is the largest rural - urban migration of this
century . Massive investments will be needed to create the jobs , housing , and infrastructure to meet soaring aspirations and make towns and cities more
livable and green . -- World Bank : India Country Overview 2013 Agriculture Main article : Agriculture in India Agriculture is an important part of the Indian
economy . At around 1,530,000 square kilometres ( 590,000 sq mi ) , India has the second - largest amount of arable land , after the US , with 52 % of total
land under cultivation . Although the total land area of the country is only slightly more than one third of China or the US , India 's arable land is
marginally smaller than that of the US , and marginally larger than that of China . However , agricultural output lags far behind its potential . The low
productivity in India is a result of several factors .
</item 9>

<item 10>
Agriculture in India
India 's integration into the global economy has been accompanied by impressive economic growth that has brought significant economic and social benefits to the
country . Nevertheless , disparities in income and human development are on the rise . Preliminary estimates suggest that in 2009 - 10 the combined all India
poverty rate was 32 % compared to 37 % in 2004 - 05 . Going forward , it will be essential for India to build a productive , competitive , and diversified
agricultural sector and facilitate rural , non-farm entrepreneurship and employment . Encouraging policies that promote competition in agricultural marketing
will ensure that farmers receive better prices . '' -- World Bank : `` India Country Overview 2011 '' A 2003 analysis of India 's agricultural growth from 1970
to 2001 by the Food and Agriculture Organisation identified systemic problems in Indian agriculture .
</item 10>
=== END ITEM id=7266e0ceb4c17edf ===

=== ITEM id=d3993ab874b1f297 ===
Gold sentences and facts:
Sentence 0: The United States two - dollar bill ( $2 ) is a current denomination of U.S. currency . The third U.S. President ( 1801 -- 09 ) , Thomas Jefferson , is featured on the obverse of the note . The reverse features an engraving of the painting The Declaration of Independence by John Trumbull . Throughout the $2 bill 's pre-1929 life as a large - sized note , it was issued as a United States Note , National Bank Note , silver certificate , Treasury or `` Coin '' Note and Federal Reserve Bank Note . When U.S. currency was changed to its current size , the $2 bill was issued only as a United States Note . Production went on until 1966 , when the series was discontinued . Ten years passed before the $2 bill was reissued as a Federal Reserve Note with a new reverse design . Two - dollar bills are seldom seen in circulation as a result of banking policies with businesses which has resulted in low production numbers due to lack of demand . This comparative scarcity in circulation , coupled with a lack of public knowledge that the bill is still in production and circulation , has also inspired urban legends and occasionally has created problems for people trying to use the bill to make purchases .
  Fact 0.0: The United States two-dollar bill ($2) is a current denomination of U.S. currency.
  Fact 0.1: Thomas Jefferson was the third U.S. President.
  Fact 0.2: Thomas Jefferson served as President from 1801 to 1809.
  Fact 0.3: Thomas Jefferson is featured on the obverse of the $2 bill.
  Fact 0.4: The reverse of the $2 bill features an engraving of the painting The Declaration of Independence by John Trumbull.
  Fact 0.5: Before 1929, the $2 bill was a large-sized note.
  Fact 0.6: Before 1929, the $2 bill was issued as a United States Note.
  Fact 0.7: Before 1929, the $2 bill was issued as a National Bank Note.
  Fact 0.8: Before 1929, the $2 bill was issued as a silver certificate.
  Fact 0.9: Before 1929, the $2 bill was issued as a Treasury or "Coin" Note.
  Fact 0.10: Before 1929, the $2 bill was issued as a Federal Reserve Bank Note.
  Fact 0.11: When U.S. currency was changed to its current size, the $2 bill was issued only as a United States Note.
  Fact 0.12: Production of the $2 bill went on until 1966, when the series was discontinued.
  Fact 0.13: Ten years passed before the $2 bill was reissued as a Federal Reserve Note with a new reverse design.
  Fact 0.14: Two-dollar bills are seldom seen in circulation as a result of banking policies with businesses.
  Fact 0.15: The banking policies with businesses resulted in low production numbers due to lack of demand.
  Fact 0.16: The comparative scarcity of $2 bills in circulation, coupled with a lack of public knowledge that the bill is still in production and circulation, has inspired urban legends.
  Fact 0.17: The comparative scarcity of $2 bills and the lack of public knowledge about them have occasionally created problems for people trying to use the bill to make purchases.

Retrieved items:
<item 1>
United States two-dollar bill
The note 's seal and serial numbers were red . The Series of 1928 $2 bill featured the treasury seal superimposed by the United States Note obligation to the
left and a large gray TWO to the right . In 1953 the $2 bill received minor design changes analogous to the $5 United States Note . The treasury seal was made
smaller and moved to the right side of the bill ; it was superimposed over the gray word TWO . The United States Note obligation now became superimposed over a
gray numeral 2 . The reverse remained unchanged . The final change to $2 United States Notes came in 1963 when the motto IN GOD WE TRUST was added to the
reverse over the Monticello . And , because dollar bills were soon to no longer be redeemable in silver , WILL PAY TO THE BEARER ON DEMAND was removed from the
obverse . Two - dollar bills , along with $5 and $100 United States Notes , were officially discontinued in August 1966 , although they remain legal tender .
1976 -- present ( edit ) Series 1976 first day of issue $2 bill with a canceled JFK postage stamp .
</item 1>

<item 2>
United States two-dollar bill
Each note was an obligation of the issuing Federal Reserve Bank and could only be redeemed at the corresponding bank . The obverse of the note featured a
borderless portrait of Thomas Jefferson to left and wording in the entire center . The reverse featured a World War I battleship . Beginning in the 1950s ,
production of $2 bills began to decrease . The relative scarcity of the bills lead some to start saving any they found , with the inevitable result that the
notes became rarer in circulation . Currently , the circulation of $2 bills accounts for around 1 % of the U.S. currency in circulation . Small size notes (
edit ) ( 6.14 × 2.61 in ≅ 156 × 66 mm ) 1928 -- 1966 ( edit ) In 1928 , when all U.S. currency was changed to its current size , the $2 bill was issued only as
a United States Note . The obverse featured a cropped version of Thomas Jefferson 's portrait that had been on previous $2 bills . The reverse featured
Jefferson 's home , Monticello . The note 's seal and serial numbers were red .
</item 2>

<item 3>
United States two-dollar bill
Although they are generally available at most banks , two - dollar bills are usually not handed out arbitrarily except upon specific request by the customer ,
and even then , may cause a delay with a trip to the vault . Rarity ( edit ) The seeming rarity of a $2 bill can be attributed to its low printing numbers as a
Federal Reserve Note . Hoarding of the series due to lack of public knowledge of the $2 bill has resulted in very few bills seen in circulation . After its
initial release , supplies of the Series 1976 two - dollar bill were allowed to dwindle until August 1996 when a new series dated 1995 began to be printed .
This series was only printed for the Federal Reserve Bank of Atlanta . Today , there is a common misconception by the general public that the $2 bill is no
longer being produced . According to the Treasury , it `` receives many letters asking why the $2 bill is no longer in circulation '' .
</item 3>

<item 4>
United States two-dollar bill
In 1976 production of the two - dollar denomination was resumed and the two - dollar bill was finally assigned as a Federal Reserve Note , with a new reverse
design featuring John Trumbull 's depiction of the drafting of the United States Declaration of Independence replacing the previous design of Monticello . It
has remained a current denomination since then . It was estimated that if the two - dollar bill replaced approximately half of the one - dollar bills in
circulation , the federal government would be able to save about $26 million in 1976 dollars ( $112 million adjusted for inflation ) over the period from 1976
to 1981 , due to reduced production , storage , and shipping costs . However , due to their limited use , two - dollar bills are not printed as frequently in a
new series like other denominations , which are produced according to demand . Though some cash registers accommodate the bill , its slot is often used for
things like checks and rolls of coins . Some bill acceptors found in vending machines , self checkout lanes , and other automated kiosks are configured to
accommodate two - dollar bills , even if the fact is not stated on the label .
</item 4>

<item 5>
United States two-dollar bill
Federal Reserve Bank of Minneapolis . Retrieved January 2 , 2018 . Jump up ^ Stone , Suzanne J. ( March -- April 1976 ) . `` The $2 Bill Returns '' ( PDF ) .
The Economic Review . Federal Reserve Bank of Richmond . 62 ( 2 ) . Retrieved December 21 , 2014 . Jump up ^ `` $2.00 still printed ? '' . Ustreas.gov .
Archived from the original on July 25 , 2010 . Retrieved July 27 , 2010 . Jump up ^ `` $2 accepting vending machines '' . 4mega-vending.com . Retrieved February
9 , 2011 . Jump up ^ `` Use The $2 '' . Retrieved October 30 , 2008 . ^ Jump up to : `` Series 2003A $2 '' . USpapermoney.info . Retrieved February 9 , 2011 .
Jump up ^ `` FAQs : Denominations of currency '' . United States Department of the Treasury . Archived from the original on August 25 , 2014 .
</item 5>

<item 6>
United States two-dollar bill
1 Large size 3.3. 2 Small size 4 Visual snapshot 5 Usage 5.1 Lockheed U-2 5.2 Incidents 6 Uncut currency sheets 7 References 8 External links Denomination
overview ( edit ) The denomination of two dollars was authorized under a congressional act , and first used in March 1862 . The denomination was continuously
used until the 1960s ; by this time the United States Note was the only remaining class of U.S. currency the two - dollar bill was assigned to . In 1966 it was
decided to discontinue all United States Notes from production , which included the two - dollar bill . The two - dollar bill initially was not reassigned to
the Federal Reserve Note class of United States currency and was thus fully discontinued ; the Treasury Department cited the two - dollar bill 's low use and
unpopularity as the reason for not immediately resuming use of the denomination .
</item 6>

<item 7>
United States two-dollar bill
In response , the Treasury stated : `` The $2 bill remains one of our circulating currency denominations ... As of April 30 , 2007 there were $1,549,052,714
worth of $2 bills in circulation worldwide . '' Things such as unusual serial numbers ( example : A11111111A ) , and replacement notes designated by a star in
the serial number can raise the collector value . `` Collectible '' two - dollar bills have been made and sold by coin dealers and others in recent years merely
by adding colors and special graphics to regular issue bills by using computer printers . However , these bills are not authorized by the Bureau of Engraving
and Printing ( BEP ) and are not worth anything more than face value on the collectors ' market . Certain conventions and tourism / convention bureaus
capitalize on the scarcity of $2 bills in circulation , encouraging convention attendees and tourists to spend $2 bills in order to illustrate to the host
communities the economic impact that the conventions and tourism bring . Sometimes known as `` SpendTom '' campaigns , the $2 bills linger in the community as a
constant reminder . Some campaigns encourage people to participate in a hunt for the bills in order to win prizes .
</item 7>

<item 8>
United States two-dollar bill
On April 13 , 1976 , the Treasury Department reintroduced the $2 bill as a cost - saving measure . Series 1976 $2 bills were redesigned and issued as a Federal
Reserve Note . The obverse design remains basically unchanged since 1928 and features the same portrait of Jefferson . A green treasury seal and serial numbers
replace the red used on the previous United States Notes . Since the reissue of the bill coincided with the United States Bicentennial , it was decided to use a
bicentennial themed design on the reverse . An engraved rendition ( not an exact reproduction ) of John Trumbull 's The Declaration of Independence replaced
Monticello on the reverse . First day issues of the new $2 bills could be taken to a post office and stamped with the date `` APR 13 1976 '' . In all ,
590,720,000 notes from Series 1976 were printed . Currently , stamped series 1976 $2 bills typically trade for about twice their face value . If the bills were
stamped in an exotic place , the value may be slightly higher .
</item 8>

<item 9>
United States two-dollar bill
When U.S. currency was changed to its current size , the $2 bill was issued only as a United States Note . Production went on until 1966 , when the series was
discontinued . Ten years passed before the $2 bill was reissued as a Federal Reserve Note with a new reverse design . Two - dollar bills are seldom seen in
circulation as a result of banking policies with businesses which has resulted in low production numbers due to lack of demand . This comparative scarcity in
circulation , coupled with a lack of public knowledge that the bill is still in production and circulation , has also inspired urban legends and occasionally
has created problems for people trying to use the bill to make purchases . Contents ( hide ) 1 Denomination overview 2 Rarity 3 History 3.1 Large - sized notes
3.2 Small size notes 3.2. 1 1928 -- 1966 3.2. 2 1976 -- present 3.3 Series dates 3.3. 1 Large size 3.3.
</item 9>

<item 10>
United States two-dollar bill
Retrieved February 9 , 2011 . CS1 maint : Uses authors parameter ( link ) Jump up ^ Stone , Suzanne J. ( March -- April 1976 ) . `` The $2 Bill Returns '' .
Retrieved February 9 , 2011 . Jump up ^ Bureau of Engraving and Printing . `` Annual Production Figures '' . Archived from the original on March 4 , 2007 .
Retrieved April 14 , 2007 . Jump up ^ `` Series 2009 $2 '' . USpapermoney.info . Retrieved April 29 , 2012 . Jump up ^ `` 2012 BEP Production Info via FOIA '' .
WheresGeorge.com . Retrieved April 29 , 2012 . Jump up ^ `` Series 2013 $2 '' . USpapermoney.info . Retrieved February 19 , 2014 . Jump up ^ `` 1869 $2 Legal
Tender Rainbow Note '' . usrarecurrency.com . Retrieved November 14 , 2015 . Jump up ^ Kathy Lawrence ( May 19 , 2011 ) .
</item 10>
=== END ITEM id=d3993ab874b1f297 ===

=== ITEM id=7060f343bff4241c ===
Gold sentences and facts:
Sentence 0: After their plane crashes in the woods , Dr. Meredith Grey ( Ellen Pompeo ) , Dr. Lexie Grey ( Chyler Leigh ) , Dr. Cristina Yang ( Sandra Oh ) , Dr. Arizona Robbins ( Jessica Capshaw ) , Dr. Derek Shepherd ( Patrick Dempsey ) , and Dr. Mark Sloan ( Eric Dane ) desperately fight to stay alive . Meredith is relatively unscathed , while the rest have serious injuries : the pilot , Jerry ( James LeGros ) , has a major spine injury , and Yang dislocates her arm . Robbins ' femur is broken and sticking through the skin , Sloan has serious internal injuries ; though initially adrenaline keeps him on his feet . Shepherd is sucked out the side of the plane and awakens alone in the wood ; his mangled hand having been pushed through the door of the plane . However , none are in as bad shape as Lexie , who is crushed under a piece of the plane . While Meredith searches for Shepherd , Yang and Sloan try to move the debris off Lexie . Eventually , the two realize that they can not save her , so Sloan holds her hand while she dies , telling her that he loves her . As Sloan tells her of the life the two were meant to have together , Lexie dies with a smile on her face just as Meredith and Yang are approaching .
  Fact 0.0: After their plane crashes in the woods, six doctors desperately fight to stay alive.
  Fact 0.1: Dr. Meredith Grey is played by Ellen Pompeo.
  Fact 0.2: Dr. Lexie Grey is played by Chyler Leigh.
  Fact 0.3: Dr. Cristina Yang is played by Sandra Oh.
  Fact 0.4: Dr. Arizona Robbins is played by Jessica Capshaw.
  Fact 0.5: Dr. Derek Shepherd is played by Patrick Dempsey.
  Fact 0.6: Dr. Mark Sloan is played by Eric Dane.
  Fact 0.7: Meredith is relatively unscathed.
  Fact 0.8: The other doctors have serious injuries.
  Fact 0.9: The pilot, Jerry (played by James LeGros), has a major spine injury.
  Fact 0.10: Yang dislocates her arm.
  Fact 0.11: Robbins' femur is broken and sticking through the skin.
  Fact 0.12: Sloan has serious internal injuries.
  Fact 0.13: Initially, adrenaline keeps Sloan on his feet.
  Fact 0.14: Shepherd is sucked out the side of the plane and awakens alone in the woods.
  Fact 0.15: Shepherd's mangled hand was pushed through the door of the plane.
  Fact 0.16: No one is in as bad shape as Lexie.
  Fact 0.17: Lexie is crushed under a piece of the plane.
  Fact 0.18: While Meredith searches for Shepherd, Yang and Sloan try to move the debris off Lexie.
  Fact 0.19: Eventually, Yang and Sloan realize they cannot save Lexie.
  Fact 0.20: Sloan holds Lexie's hand while she dies.
  Fact 0.21: Sloan tells Lexie that he loves her.
  Fact 0.22: Sloan tells Lexie about the life the two were meant to have together.
  Fact 0.23: Lexie dies with a smile on her face.
  Fact 0.24: Meredith and Yang are approaching just as Lexie dies.
Sentence 1: `` Flight '' is the twenty - fourth and final episode of the eighth season of the American television medical drama Grey 's Anatomy , and the show 's 172nd episode overall . It was written by series creator Shonda Rhimes , and directed by Rob Corn . The episode was originally broadcast on the American Broadcasting Company ( ABC ) in the United States on May 17 , 2012 . In the episode , six doctors from Seattle Grace Mercy West Hospital who are victims of an aviation accident fight to stay alive , but Dr. Lexie Grey ( Chyler Leigh ) ultimately dies . Other storylines occur in Seattle where Dr. Richard Webber ( James Pickens , Jr . ) plans his annual dinner for the departing residents , Dr. Owen Hunt ( Kevin McKidd ) fires Dr. Teddy Altman ( Kim Raver ) , and Dr. Miranda Bailey ( Chandra Wilson ) gets engaged .
  Fact 1.0: 'Flight' is the twenty-fourth and final episode of the eighth season of the American television medical drama Grey's Anatomy.
  Fact 1.1: 'Flight' is the show's 172nd episode overall.
  Fact 1.2: 'Flight' was written by series creator Shonda Rhimes.
  Fact 1.3: 'Flight' was directed by Rob Corn.
  Fact 1.4: 'Flight' was originally broadcast on the American Broadcasting Company (ABC) in the United States on May 17, 2012.
  Fact 1.5: In the episode, six doctors from Seattle Grace Mercy West Hospital who are victims of an aviation accident fight to stay alive.
  Fact 1.6: In the episode, Dr. Lexie Grey (played by Chyler Leigh) ultimately dies.
  Fact 1.7: Other storylines occur in Seattle.
  Fact 1.8: Dr. Richard Webber (played by James Pickens, Jr.) plans his annual dinner for the departing residents.
  Fact 1.9: Dr. Owen Hunt (played by Kevin McKidd) fires Dr. Teddy Altman (played by Kim Raver).
  Fact 1.10: Dr. Miranda Bailey (played by Chandra Wilson) gets engaged.

Retrieved items:
<item 1>
Flight (Grey's Anatomy)
Grey 's Anatomy has once again managed to shock with its season finale . '' While she appreciated the `` realism and authenticity that Grey 's is known for '' ,
she found the episode was `` almost too much '' as it was `` extremely gory and difficult to watch , initially because of the grisly wounds '' but later because
of the `` heavy and emotional things that transpired '' . She thought Pompeo gave one of her best performances when her character learned that her sister was
dead . Digital Spy 's Ben Lee found Leigh 's and Dane 's performances `` phenomenal '' and added that he had probably never seen a better performance from Dane
. He described the moment the two actors shared as `` truly poignant '' . To him , Lexie 's death felt like a finale , which was thus too early . As for what
was happening in Seattle Grace , he thought it was `` a bit pointless '' and `` uninteresting '' except Altman 's departure , which he deemed `` the most
significant moment at the hospital '' . Entertainment Weekly 's Tanner Stransky commented of Lexie 's death : `` It was an intense death .
</item 1>

<item 2>
Flight (Grey's Anatomy)
Lexie 's death devastates Meredith , who is still desperately trying to find her husband . Eventually , she and Shepherd reunite and they fix his hand as best
as they can . Meanwhile , back at Seattle Grace Mercy West Hospital , no one is aware of what has happened to the other doctors . Dr. Richard Webber ( James
Pickens , Jr . ) prepares the annual dinner for the departing residents , which Dr. Alex Karev ( Justin Chambers ) , Dr. Jackson Avery ( Jesse Williams ) , and
Dr. April Kepner ( Sarah Drew ) are dreading . Avery makes the choice to take a job offer at Tulane Medical Center , and he and Kepner share a moment . Dr. Ben
Warren ( Jason George ) and Dr. Miranda Bailey ( Chandra Wilson ) decide to get married , even though Warren is going to start his surgical internship in Los
Angeles . After realizing Dr. Teddy Altman ( Kim Raver ) was offered a chief position at United States Army Medical Command ( MEDCOM ) and is refusing to leave
Seattle out of loyalty , Dr.
</item 2>

<item 3>
Flight (Grey's Anatomy)
Everybody was very accommodating -- the crew , cast . And I opted to stay underneath ( the wreckage ) for the most part over two days rather than trying to get
in and out . '' In regard to the episode , Rhimes commented before it originally aired that it was difficult to write , largely because of the death of a main
character . She compared it to writing the season six finale , by explaining that the former was `` more painful '' to write . After the episode aired , Rhimes
repeated in a tweet that it was hard for her to write the finale , adding : `` I did not enjoy it . It made me sick and it made me sad . '' Rhimes also
explained the departure of Leigh , whose character died after the plane crash , by saying that the two came to an agreement on the decision to kill Lexie ,
after extensive discussion . Speaking of Raver 's departure whose character left Seattle Grace for MEDCOM , Rhimes elaborated that Raver was offered a contract
renewal , but declined . Reception ( edit ) Chyler Leigh 's performance was deemed `` phenomenal '' .
</item 3>

<item 4>
Flight (Grey's Anatomy)
I mean , how awful was it to watch one of Grey 's longest - running characters pass away so quickly -- and rather unceremoniously ? I get that Ms. Rhimes had to
do what she had to do -- and every show needs to be shaken up once in a while -- but I do n't love that Lexie was the one to die . Could it have been someone
less important somehow ? I guess it would have been too obvious to do Kepner . And you probably just do n't kill off a hottie like Mark Sloan , right ? ''
Stransky also complained about Robbins ' screams at the beginning of the episode , but enjoyed Bailey 's story . In an Entertainment Weekly poll that judged all
the television season finales of the year , Lexie 's death was voted the `` Top Tissue Moment '' , while Robbins ' injured leg and Shepherd 's mangled hand were
voted the `` Most Disturbing Image '' . The ending of the episode was also considered as the `` Best Ending to an Otherwise So - So Season '' .
</item 4>

<item 5>
Flight (Grey's Anatomy)
In the episode , six doctors from Seattle Grace Mercy West Hospital who are victims of an aviation accident fight to stay alive , but Dr. Lexie Grey ( Chyler
Leigh ) ultimately dies . Other storylines occur in Seattle where Dr. Richard Webber ( James Pickens , Jr . ) plans his annual dinner for the departing
residents , Dr. Owen Hunt ( Kevin McKidd ) fires Dr. Teddy Altman ( Kim Raver ) , and Dr. Miranda Bailey ( Chandra Wilson ) gets engaged . The episode marked
Leigh 's and Raver 's final appearance to the series . Exterior filming of the accident took place at Big Bear Lake , California . Jason George reprised his
role as a guest star , whereas James LeGros made his first appearance . The episode opened to mixed reviews from television critics , with some criticizing the
death of Lexie , but praising Leigh 's performance , in addition to Ellen Pompeo ( Dr. Meredith Grey ) 's and Eric Dane ( Dr. Mark Sloan ) 's .
</item 5>

<item 6>
Flight (Grey's Anatomy)
Mark Sloan ( Eric Dane ) desperately fight to stay alive . Meredith is relatively unscathed , while the rest have serious injuries : the pilot , Jerry ( James
LeGros ) , has a major spine injury , and Yang dislocates her arm . Robbins ' femur is broken and sticking through the skin , Sloan has serious internal
injuries ; though initially adrenaline keeps him on his feet . Shepherd is sucked out the side of the plane and awakens alone in the wood ; his mangled hand
having been pushed through the door of the plane . However , none are in as bad shape as Lexie , who is crushed under a piece of the plane . While Meredith
searches for Shepherd , Yang and Sloan try to move the debris off Lexie . Eventually , the two realize that they can not save her , so Sloan holds her hand
while she dies , telling her that he loves her . As Sloan tells her of the life the two were meant to have together , Lexie dies with a smile on her face just
as Meredith and Yang are approaching . Lexie 's death devastates Meredith , who is still desperately trying to find her husband .
</item 6>

<item 7>
Flight (Grey's Anatomy)
Mark Sloan ) 's . `` Flight '' earned Rhimes an NAACP Image Award nomination and it was also nominated under several categories of Entertainment Weekly 's
finale awards . Upon its initial airing , the episode was viewed in the United States by 11.44 million people , garnered a 4.1 / 11 Nielsen rating / share in
the 18 -- 49 demographic , ranking fourth for the night in terms of viewership , and registering as Thursday 's highest - rated drama . Contents ( hide ) 1 Plot
2 Production 3 Reception 4 References 5 External links Plot ( edit ) After their plane crashes in the woods , Dr. Meredith Grey ( Ellen Pompeo ) , Dr. Lexie
Grey ( Chyler Leigh ) , Dr. Cristina Yang ( Sandra Oh ) , Dr. Arizona Robbins ( Jessica Capshaw ) , Dr. Derek Shepherd ( Patrick Dempsey ) , and Dr. Mark Sloan
( Eric Dane ) desperately fight to stay alive .
</item 7>

<item 8>
Flight (Grey's Anatomy)
Owen Hunt ( Kevin McKidd ) , the chief of surgery , fires her to free her from the hospital where her husband died . As the episode ends , Hunt picks up his
messages to discover the surgical team never made it to Boise . The residents , finally excited to celebrate at Webber 's dinner , are left waiting for their
stranded friends . The remaining crash survivors are left struggling to stay awake as their last match goes out . In the closing monologue , Meredith repeats
the opening to the speech that Webber gave in the pilot episode of the series . Production ( edit ) `` It was so hard . It was physically and emotionally
challenging because you 're constantly at extremely high stakes . The physicality for Cristina , she had to do a lot of things with one arm , and I had to do a
lot of things with one shoe . You 're running around the mountain with one shoe and one arm basically and doing things that way . And it 's like we 've had this
cushy time on stages and in the hospital . But being out in the open on the side of a mountain was exhilarating ; you can not beat location work .
</item 8>

<item 9>
Flight (Grey's Anatomy)
`` ' Grey 's Anatomy ' recap : Major death in season finale ' Flight ' '' . Digital Spy . Nat Mags . Retrieved May 22 , 2012 . Jump up ^ Stransky , Tanner ( May
18 , 2012 ) . `` ' Grey 's Anatomy ' season finale recap : A Grey Goes Down '' . Entertainment Weekly . Time Inc . Retrieved May 22 , 2012 . Jump up ^ Bierly ,
Mandi ( May 29 , 2012 ) . `` Grey 's Anatomy , Chyler Leigh , ... TV Season Finale Awards : YOUR 2012 Winners ! Photo 9 of 20 '' . Entertainment Weekly . Time
Inc . Retrieved June 17 , 2012 . Jump up ^ Bierly , Mandi ( May 29 , 2012 ) . `` Grey 's Anatomy , Ellen Pompeo TV Season Finales TV Season Finale Awards : YOUR
2012 Winners ! Photo 12 of 20 '' . Entertainment Weekly . Time Inc . Retrieved June 18 , 2012 .
</item 9>

<item 10>
Flight (Grey's Anatomy)
Jump up ^ Writer : Rhimes , Shonda . Director : Corn , Rob ( May 17 , 2012 ) . `` Flight '' . Grey 's Anatomy . Season 8 . American Broadcasting Company . Jump
up ^ `` Season 8 , Episode : 24 `` Flight '' Music `` . TV Fanatic . SheKnows Entertainment . Retrieved July 22 , 2012 . Jump up ^ Dos Santos , Kristin ( May 1
, 2012 ) . `` Spoiler Chat Daily : American Horror Story Season Two Scoop ! Plus , Grey 's , True Blood and More ! '' . E ! Online . NBCUniversal . Retrieved
July 21 , 2012 . Jump up ^ Hilton , Perez ( June 24 , 2010 ) . `` Chyler Leigh Describes Filming Deets Behind Grey 's Anatomy Death ! '' . PerezHilton.com .
Retrieved July 21 , 2012 . Jump up ^ Goldberg , Lesley ( April 26 , 2012 ) .
</item 10>
=== END ITEM id=7060f343bff4241c ===
